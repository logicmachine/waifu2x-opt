#ifndef WAIFU2X_H
#define WAIFU2X_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#	ifdef WAIFU2X_BUILD_SHARED_LIBRARY
#		define W2X_EXPORT __declspec(dllexport)
#	else
#		define W2X_EXPORT __declspec(dllimport)
#	endif
#else
#	define W2X_EXPORT
#endif

typedef void * W2xHandle;

W2X_EXPORT W2xHandle w2x_create_handle(const char *model_json);
W2X_EXPORT void w2x_destroy_handle(W2xHandle handle);

W2X_EXPORT int w2x_set_num_threads(W2xHandle handle, int num_threads);
W2X_EXPORT int w2x_set_block_size(W2xHandle handle, int width, int height);

W2X_EXPORT int w2x_get_num_steps(const W2xHandle handle);

W2X_EXPORT int w2x_process(
	W2xHandle handle, float *dst, const float *src,
	int width, int height, int pitch, int verbose);

#ifdef __cplusplus
}
#endif

#endif

