#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

EXTERN int legate_mpi_init(int*, char***);
EXTERN int legate_mpi_finalize(void);

// a short little script to make sure that the wrapper works/linked properly
int main(int argc, char* argv[])
{
  int ret = 0;

  ret = legate_mpi_init(&argc, &argv);
  if (ret) {
    return ret;
  }
  return legate_mpi_finalize();
}
