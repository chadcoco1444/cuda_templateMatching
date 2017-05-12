// OpenCV library
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "opencv2/gpu/gpu.hpp"

#define VALUE_MAX 10000000

#define CPU
//#define DEBUG

struct match {

	int diffRow;
	int diffCol;
	int diff;

} pos;

// CPU Function define 
int *CPU_ImgMatching(IplImage* sourceImg, IplImage* patternImg);
void CPU_ImgFindDiff(IplImage* sourceImg,  IplImage* patternImg, int *host_result);
// GPU Function define
void GPU_ImgMatching(IplImage* sourceImg, IplImage* patternImg);
__global__ void GPU_Kernel_ImgMatching(unsigned char * d_ImgSrc ,unsigned char *d_pImgSrc,int *d_diffDst,int Width,int Height, int pWidth, int pHeight);


int main(int argc, char *argv[]){

	IplImage* sourceImg; 
	IplImage* patternImg;

	sourceImg = cvLoadImage("lena.jpg", 0);
	patternImg = cvLoadImage("lena_eye.jpg", 0);

#ifdef DEBUG

	printf("Height\tof\tsourceImg:%d\n",sourceImg->height);
	printf("Width\tof\tsourceImg:%d\n",sourceImg->width);
	printf("Size\tof\tsourceImg:%d\n\n",sourceImg->imageSize);
	printf("Height\tof\tpatternImg:%d\n",patternImg->height);
	printf("Width\tof\tpatternImg:%d\n",patternImg->width);
	printf("Size\tof\tpatternImg:%d\n\n",patternImg->imageSize);

#endif

#ifdef CPU

	struct timespec t_start, t_end;
	double elapsedTime;
	
	clock_gettime( CLOCK_REALTIME, &t_start);

// CPU kernel
	int *host_result = CPU_ImgMatching(sourceImg, patternImg);
// CPU Find Diff
	CPU_ImgFindDiff(sourceImg, patternImg, host_result);

	clock_gettime( CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("CPU time:\t%13f ms\n", elapsedTime);

#endif	
// GPU invoke function
	GPU_ImgMatching(sourceImg, patternImg);

	cvWaitKey(0); 
	cvDestroyWindow( "sourceImage" );
	cvReleaseImage( &sourceImg );
	cvDestroyWindow( "patternImage" );
	cvReleaseImage( &patternImg );
	
return 0;

}



__global__ void GPU_Kernel_ImgMatching(unsigned char * d_ImgSrc ,unsigned char *d_pImgSrc,int *d_diffDst,int Width,int Height, int pWidth, int pHeight)
{
	     
 	int diff;
 	int result_height = Height - pHeight + 1;
	int result_width  = Width  - pWidth  + 1;
   
  	uchar p_sourceIMG, p_patternIMG;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

  	if(tid < Width) {
  		for(int row = 0; row < result_height; row++ ) {
    		diff = 0;
    		for(int i=0; i<pHeight; i++) {            
        		for(int j=0; j<pWidth; j++) {               
					p_sourceIMG = d_ImgSrc[(row + i) * Width + tid + j];
					p_patternIMG = d_pImgSrc[i * pWidth + j];
					if(p_sourceIMG > p_patternIMG) diff += p_sourceIMG - p_patternIMG;
					else diff += p_patternIMG - p_sourceIMG;
				}       
    		} 
    		d_diffDst[row * result_width + tid] = diff;
    	}
    	
    }

}

void GPU_ImgMatching(IplImage* sourceImg, IplImage* patternImg)
{

	int Width = sourceImg->width;
	int Height = sourceImg->height;
	int pWidth = patternImg->width;
	int pHeight = patternImg->height;

// GPU Size
	size_t d_sizeDiff = sizeof(int) * (Width - pWidth + 1) * (Height - pHeight + 1) ;
	size_t d_sizeImg  = sizeof(unsigned char) * Width * Height;
	size_t d_psizeImg = sizeof(unsigned char) * pWidth * pHeight;
	
// CPU
	unsigned char *h_ImgSrc  = (unsigned char*)(sourceImg->imageData);
	unsigned char *h_pImgSrc = (unsigned char*)(patternImg->imageData);
	int *h_diffDst = (int *)malloc(d_sizeDiff);

// GPU 
	int *d_diffDst 			 = NULL;
	unsigned char *d_ImgSrc  = NULL;
	unsigned char *d_pImgSrc = NULL;
// GPU memory allocation
	cudaMalloc((void**)&d_diffDst, d_sizeDiff);
	cudaMalloc((void**)&d_ImgSrc , d_sizeImg);
	cudaMalloc((void**)&d_pImgSrc, d_psizeImg);
// Memory copy from host to device
	cudaMemcpy(d_diffDst, h_diffDst, d_sizeDiff, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pImgSrc, h_pImgSrc, d_psizeImg, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgSrc , h_ImgSrc , d_sizeImg , cudaMemcpyHostToDevice);
// define the block and thread
    dim3 dimGrid(8);
    dim3 dimBlock(128);
    
// Cuda time profile
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
// the kernel function
    GPU_Kernel_ImgMatching<<<dimGrid, dimBlock>>>(d_ImgSrc, d_pImgSrc, d_diffDst, Width, Height, pWidth, pHeight);

// Handle the Kernel function error
    cudaError_t cuda_err = cudaGetLastError();
    if ( cudaSuccess != cuda_err ){
        printf("before kernel call: error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time:\t%13f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
// Memory copy from device to host  
    cudaMemcpy(h_diffDst, d_diffDst, d_sizeDiff, cudaMemcpyDeviceToHost);

// Call CPU Find Diff function
    CPU_ImgFindDiff(sourceImg, patternImg, h_diffDst);
    CvPoint pt1, pt2;

	pt1.x = pos.diffCol;
	pt1.y = pos.diffRow;
	pt2.x = pt1.x + patternImg->width;
	pt2.y = pt1.y + patternImg->height;

	// Draw the rectangle in the source image
	cvRectangle( sourceImg, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
	cvNamedWindow( "sourceImage", 1 );
	cvShowImage( "sourceImage", sourceImg );
	cvNamedWindow( "patternImage", 1 );
	cvShowImage( "patternImage", patternImg );


	cudaFree(d_diffDst);
	cudaFree(d_pImgSrc);
	cudaFree(d_ImgSrc);
	
}

int *CPU_ImgMatching(IplImage* sourceImg, IplImage* patternImg){

	int diff;
	int result_height = sourceImg->height - patternImg->height + 1;
	int result_width  = sourceImg->width  - patternImg->width  + 1;

	int *host_result;
	host_result = (int *)malloc(result_height * result_width * sizeof(int));
	uchar p_sourceIMG, p_patternIMG;

	for(int row = 0; row < result_height; row++ ) {
		for( int col = 0; col < result_width; col++ ) {
			diff = 0.0;
				for ( int j = 0; j < patternImg->height; j++ ){
					for ( int i = 0; i < patternImg->width; i++ ) {
						p_sourceIMG = sourceImg->imageData[(row + j) * sourceImg->widthStep+col+i];
						p_patternIMG = patternImg->imageData[j * patternImg->widthStep+i];
						diff += abs( p_sourceIMG - p_patternIMG );
					}
				}
				host_result[row * result_width + col] = diff;
		}
	}
	return host_result;

}

void CPU_ImgFindDiff(IplImage* sourceImg, IplImage* patternImg, int *host_result){
	
	int minDiff = VALUE_MAX;

	int result_height = sourceImg->height - patternImg->height + 1;
	int result_width  = sourceImg->width  - patternImg->width  + 1;

	for( int row = 0; row < result_height; row++ ) {
		for( int col = 0; col < result_width; col++ ) {
			if ( minDiff > host_result[row * result_width + col] ) {
				minDiff = host_result[row * result_width + col];

				pos.diffRow = row;
				pos.diffCol = col;
				pos.diff = host_result[row * result_width + col];
			}
		}
	}
#ifdef DEBUG
	printf("minSAD:%d\n",minDiff);
#endif
	free(host_result);
}


