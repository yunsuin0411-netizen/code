// bmp.c : 24bit BMP 구조체+2D 배열로 읽기/쓰기
#include <stdio.h>
#include <stdlib.h>
#include "bmp.h"
#include "image.h"

// 한 줄에 필요한 바이트 수 (픽셀은 3바이트, 4바이트 배수로 패딩)
static int rowsize(int width) {
    return (width * 3 + 3) & (~3); // 4바이트 패딩
}

// BMP 파일 읽기 함수
Image* loadBMP_Image(char* filename, BITMAPFILEHEADER* fileHeader, BITMAPINFOHEADER* infoHeader) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { 
        printf("파일 열기 실패: %s\n", filename); 
        return NULL; 
    }

    // 파일 헤더와 정보 헤더 읽기
    if (fread(fileHeader, sizeof(BITMAPFILEHEADER), 1, fp) != 1) { 
        fclose(fp); 
        return NULL; 
    }
    if (fread(infoHeader, sizeof(BITMAPINFOHEADER), 1, fp) != 1) { 
        fclose(fp); 
        return NULL; 
    }

    // BMP 파일인지, 24bit인지, 압축 없는지 확인
    if (fileHeader->bfType != 0x4D42) { 
        printf("BMP 파일이 아닙니다.\n"); 
        fclose(fp); return NULL; 
    }
    if (infoHeader->biBitCount != 24) { 
        printf("24비트 BMP만 지원합니다.\n"); 
        fclose(fp); return NULL; 
    }
    if (infoHeader->biCompression != 0) { 
        printf("압축 BMP는 지원하지 않습니다.\n"); 
        fclose(fp); 
        return NULL; 
    }

    int w = infoHeader->biWidth;
    int h = infoHeader->biHeight;
    int height = (h < 0) ? -h : h;     // 음수면 top-down, 양수면 bottom-up
    int row = rowsize(w);  // 한 줄 저장 크기

    // 이미지 구조체 동적할당
    Image* img = allocImage(w, height);
    if (!img) { 
        printf("메모리 할당 실패\n"); 
        fclose(fp); 
        return NULL; 
    }

    // 픽셀 데이터 시작 위치로 이동
    fseek(fp, fileHeader->bfOffBits, SEEK_SET);

    // 픽셀 읽기
    for (int y = 0; y < height; y++) {
        // bottom-up 저장이면 뒤집어서 넣기
        int yy = (h > 0) ? (height - 1 - y) : y;
        for (int x = 0; x < w; x++) {
            unsigned char bgr[3];
            if (fread(bgr, 1, 3, fp) != 3) {
                printf("픽셀 읽기 실패\n");
                freeImage(img); fclose(fp); 
                return NULL;
            }
            img->data[yy][x].b = bgr[0];
            img->data[yy][x].g = bgr[1];
            img->data[yy][x].r = bgr[2];
        }
        // 줄 끝의 패딩 건너뛰기
        int pad = row - w * 3;
        if (pad > 0) 
            fseek(fp, pad, SEEK_CUR);
    }

    fclose(fp);
    return img;
}

// BMP 파일 저장 함수 (항상 bottom-up으로 저장)
int saveBMP_Image(char* filename, BITMAPFILEHEADER* fhIn, BITMAPINFOHEADER* ihIn, Image* img) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { 
        printf("파일 저장 실패: %s\n", filename); 
        return 0; 
    }

    int w = img->width;
    int h = img->height;
    int row = rowsize(w);
    int pad = row - w * 3;

    // 헤더 값 복사 후 안전하게 초기화
    BITMAPFILEHEADER fh = *fhIn;
    BITMAPINFOHEADER ih = *ihIn;

    fh.bfType      = 0x4D42; // "BM"
    fh.bfReserved1 = 0;
    fh.bfReserved2 = 0;
    fh.bfOffBits   = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

    ih.biSize        = sizeof(BITMAPINFOHEADER);
    ih.biWidth       = w;
    ih.biHeight      = h;      // 양수 → bottom-up
    ih.biPlanes      = 1;
    ih.biBitCount    = 24;
    ih.biCompression = 0;      // 무압축
    ih.biSizeImage   = row * h;
    if (ih.biXPelsPerMeter == 0) 
        ih.biXPelsPerMeter = 2835; // 72 DPI 기본값
    if (ih.biYPelsPerMeter == 0) 
        ih.biYPelsPerMeter = 2835;
    ih.biClrUsed      = 0;
    ih.biClrImportant = 0;

    fh.bfSize = fh.bfOffBits + ih.biSizeImage;

    // 헤더 쓰기
    if (fwrite(&fh, sizeof(fh), 1, fp) != 1) { 
        fclose(fp); 
        return 0; 
    }
    if (fwrite(&ih, sizeof(ih), 1, fp) != 1) { 
        fclose(fp); 
        return 0; 
    }

    // 패딩 영역(0으로 채움)
    unsigned char* padding = (pad > 0) ? (unsigned char*)calloc(pad, 1) : NULL;

    // bottom-up 방식으로 아래 줄부터 기록
    for (int y = h - 1; y >= 0; y--) {
        for (int x = 0; x < w; x++) {
            unsigned char bgr[3] = { img->data[y][x].b, img->data[y][x].g, img->data[y][x].r };
            if (fwrite(bgr, 1, 3, fp) != 3) {
                if (padding) 
                    free(padding);
                fclose(fp); 
                return 0;
            }
        }
        if (pad > 0) 
            fwrite(padding, 1, pad, fp);
    }

    if (padding) 
        free(padding);
    fclose(fp);
    return 1;
}
