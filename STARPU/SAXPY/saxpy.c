#include "starpu.h"
#include <stdio.h>

#define SIZE 1024

void saxpy_cpu(void *buffers[], void *_args)
{
  float *factor = (float *)_args;
  struct starpu_vector_interface *y = (struct starpu_vector_interface *)buffers[0];
  struct starpu_vector_interface *x = (struct starpu_vector_interface *)buffers[1];

  unsigned size = STARPU_VECTOR_GET_NX(x);

  float *x_v = (float *)STARPU_VECTOR_GET_PTR(x);
  float *y_v = (float *)STARPU_VECTOR_GET_PTR(y);

  float a = *factor;
  for (unsigned i = 0; i < size; ++i)
  {
    y_v[i] = x_v[i] * a + y_v[i];
  }
}

struct starpu_codelet cl = {.where = STARPU_CPU,
                            .cpu_funcs = {saxpy_cpu},
                            .cpu_funcs_name = {"saxpy_cpu"},
                            .nbuffers = 2,
                            .modes = {STARPU_RW, STARPU_R}};

int main(int argc, char *argv[]) {
  float x[SIZE];
  float y[SIZE];
  for (int i = 0; i < SIZE; ++i) {
    x[i] = 2.f * i + 50.f;
    y[i] = 0.5f * i + 100.f;
  }

  fprintf(stdout, "[BEFORE] First element: %f\n", y[0]);

  int ret = starpu_init(NULL);
  if (ret != 0) {
    return 1;
  }

  printf("%d CPU Cores\n", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));

  float scale = 5.f;
  starpu_data_handle_t x_handle, y_handle;
  starpu_vector_data_register(&x_handle, STARPU_MAIN_RAM, (uintptr_t)x, SIZE,
                              sizeof(x[0]));
  starpu_vector_data_register(&y_handle, STARPU_MAIN_RAM, (uintptr_t)y, SIZE,
                              sizeof(y[0]));

  struct starpu_task *task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &cl;
  task->handles[0] = y_handle;
  task->handles[1] = x_handle;

  task->cl_arg = &scale;
  task->cl_arg_size = sizeof(scale);

  ret = starpu_task_submit(task);
  if (ret != 0) {
    return 1;
  }
  starpu_data_unregister(x_handle);
  starpu_data_unregister(y_handle);

  starpu_shutdown();
  fprintf(stdout, "[AFTER] First element: %f\n", y[0]);
  return 0;
}
