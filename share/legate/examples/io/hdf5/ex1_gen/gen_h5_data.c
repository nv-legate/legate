/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 *  Generate HDF5 data (gen_h5_data)
 *
 *  Generate an HDF5 file with a number of 1-D datasets in it.
 *
 */

#include "hdf5.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default file name
#define GEN_H5_DEF_FILENAME "gen_data.h5"
#define GEN_H5_DEF_NDSETS 8
#define GEN_H5_DEF_DSET_ELMTS (1024 * 1024 * 1024)
#define GEN_H5_DEF_DSET_TYPE H5T_NATIVE_INT
#define GEN_H5_DEF_DSET_VALUE 0
#define GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE (1024 * 1024)
#define GEN_H5_DEF_VIRTUAL_DSET_BASE "virt_file"

typedef struct user_params {
  char* filename;                // Name of file created
  unsigned ndsets;               // Number of 1-D datasets to create in file
  hsize_t dset_elmts;            // Number of elements in each dataset
  hid_t dset_type;               // Type of dataset each element
  bool verbose;                  // Display more info during execution
  bool dset_value;               // Use dataset index for element values
  bool virtual_dsets;            // Create virtual datasets, instead of contiguous ones
  char* virtual_dset_base_name;  // If creating virtual datasets, the base file name
} user_params_t;

void user_params_destroy(user_params_t* args)
{
  free(args->filename);
  free(args->virtual_dset_base_name);
}

static void usage(const char* argv0)
{
  printf("Usage:\n");
  printf(" %s	- Generate HDF5 data\n", argv0);
  printf("\n");
  printf("Options:\n");
  printf(
    "  -e <size>, --dset_elmts=<size>\n\t\t Number of elements in datasets created\n\t\t (default: "
    "%llu)\n",
    (unsigned long long)GEN_H5_DEF_DSET_ELMTS);
  printf("  -f <name>, --filename=<name>\n\t\t Name of file to create\n\t\t (default: '%s')\n",
         GEN_H5_DEF_FILENAME);
  printf(
    "  -i <name>, --virtual=<name>\n\t\t Create <n> virtual datasets, with base path prefix "
    "<name>\n\t\t (default: '%s')\n",
    GEN_H5_DEF_VIRTUAL_DSET_BASE);
  printf(
    "  -n <count>, --num_dsets=<count>\n\t\t Number of datasets to create\n\t\t (default: %u)\n",
    GEN_H5_DEF_NDSETS);
  printf(
    "  -t <type>, --datatype=<type>\n\t\t Datatype of dataset elements, 'int', 'double', 'char' "
    "supported\n\t\t (default: 'int')\n");
  printf("  -v, --verbose\n\t\t Display verbose information\n\t\t (default: disabled)\n");
  printf(
    "  -V, --dset_value\n\t\t Write dataset index as element values\n\t\t (default: 0 element "
    "values)\n\n");
}

static int parse_command_line(int argc, char* argv[], user_params_t* args)
{
  static struct option long_options[] = {
    {"dset_elmts", required_argument, NULL, 'e'},
    {"filename", required_argument, NULL, 'f'},
    {"virtual", required_argument, NULL, 'i'},
    {"num_dsets", required_argument, NULL, 'n'},
    {"datatype", required_argument, NULL, 't'},
    {"verbose", no_argument, NULL, 'v'},
    {"dset_value", no_argument, NULL, 'V'},
    {NULL, 0, NULL, 0},
  };

  // Set defaults
  memset(args, 0, sizeof(*args));
  args->ndsets     = GEN_H5_DEF_NDSETS;
  args->dset_elmts = GEN_H5_DEF_DSET_ELMTS;
  args->dset_type  = GEN_H5_DEF_DSET_TYPE;

  while (true) {
    int c;

    c = getopt_long(argc, argv, "e:f:i:n:t:vV", long_options, NULL);
    if (c < 0) {
      break;
    }

    switch (c) {
      case 'e': args->dset_elmts = strtol(optarg, NULL, 0); break;
      case 'f': args->filename = strdup(optarg); break;
      case 'i': {
        args->virtual_dsets          = true;
        args->virtual_dset_base_name = strdup(optarg);
        args->filename               = strdup(optarg);
        break;
      }
      case 'n': args->ndsets = strtol(optarg, NULL, 0); break;
      case 't':
        if (0 == strcmp(optarg, "char")) {
          args->dset_type = H5T_NATIVE_CHAR;
        } else if (0 == strcmp(optarg, "int")) {
          args->dset_type = H5T_NATIVE_INT;
        } else if (0 == strcmp(optarg, "double")) {
          args->dset_type = H5T_NATIVE_DOUBLE;
        } else {
          fprintf(stderr, "\nERROR: invalid type name specified\n\n");
          goto error;
        }
        break;
      case 'v': args->verbose = true; break;
      case 'V': args->dset_value = true; break;
      default: goto error;
    }
  }

  if (optind < argc) {
    goto error;
  }

  // Set default filename, if none provided
  if (NULL == args->filename) {
    args->filename = strdup(GEN_H5_DEF_FILENAME);
  }

  if (NULL == args->virtual_dset_base_name) {
    args->virtual_dset_base_name = strdup(GEN_H5_DEF_VIRTUAL_DSET_BASE);
  }

  if (0 == args->ndsets) {
    fprintf(stderr, "\nERROR: please specify a non-zero number of datasets\n\n");
    goto error;
  }

  if (0 == args->dset_elmts) {
    fprintf(stderr, "\nERROR: please specify a non-zero number of dataset elements\n\n");
    goto error;
  }

  return 0;

error:
  user_params_destroy(args);
  usage(argv[0]);
  return 1;
}

int main(int argc, char* argv[])
{
  user_params_t args;
  hid_t fid;      // HDF5 file ID
  hid_t sid;      // HDF5 dataspace ID
  hid_t fapl_id;  // HDF5 file access property list
  hid_t dcpl_id;  // HDF5 dataset creation property list

  // Parse the params
  if (parse_command_line(argc, argv, &args)) {
    fprintf(stderr, "error parsing parameters\n");
    goto error;
  }

  // Print the parameters
  if (args.verbose) {
    printf("Filename: %s\n", args.filename);
    printf("Number of datasets: %u\n", args.ndsets);
    if (args.dset_elmts >= (1024 * 1024 * 1024)) {
      printf("Number of elements in each dataset: %llu Gelmts\n",
             (unsigned long long)(args.dset_elmts / (1024 * 1024 * 1024)));
    } else if (args.dset_elmts >= (1024 * 1024)) {
      printf("Number of elements in each dataset: %llu Melmts\n",
             (unsigned long long)(args.dset_elmts / (1024 * 1024)));
    } else if (args.dset_elmts >= 1024) {
      printf("Number of elements in each dataset: %llu Kelmts\n",
             (unsigned long long)(args.dset_elmts / 1024));
    } else {
      printf("Number of elements in each dataset: %llu elmts\n",
             (unsigned long long)args.dset_elmts);
    }
    if (1 == H5Tequal(args.dset_type, H5T_NATIVE_CHAR)) {
      printf("Dataset element type: 'char'\n");
    } else if (1 == H5Tequal(args.dset_type, H5T_NATIVE_INT)) {
      printf("Dataset element type: 'int'\n");
    } else if (1 == H5Tequal(args.dset_type, H5T_NATIVE_DOUBLE)) {
      printf("Dataset element type: 'double'\n");
    } else {
      printf("Dataset element type: 'unknown'\n");
    }
    printf("\n");
  }

  // File access property list
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  if (fapl_id < 0) {
    fprintf(stderr, "error creating HDF5 file access property list\n");
    goto error;
  }

  // Allocate space for 1MB of metadata at a time, to improve performance
  if (H5Pset_meta_block_size(fapl_id, (1024 * 1024)) < 0) {
    fprintf(stderr, "error setting metadata allocation size\n");
    goto error;
  }

  // Set alignment of objects > 1MB (i.e. storage for dataset elements)
  // to be on 1MB boundary, to improve I/O performance
  if (H5Pset_alignment(fapl_id, (1024 * 1024), (1024 * 1024)) < 0) {
    fprintf(stderr, "error setting file object alignment\n");
    goto error;
  }

  // Create the HDF5 file
  fid = H5Fcreate(args.filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  if (fid < 0) {
    fprintf(stderr, "error creating HDF5 file\n");
    goto error;
  }

  // Dataset creation property list
  dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  if (dcpl_id < 0) {
    fprintf(stderr, "error creating HDF5 dataset creation property list\n");
    goto error;
  }

  // Set allocation time to 'early', so space in the file is reserved for
  // each dataset
  if (H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_EARLY) < 0) {
    fprintf(stderr, "error setting dataset allocation time\n");
    goto error;
  }

  // Check for virtual datasets
  if (args.virtual_dsets) {
    hsize_t dims = args.dset_elmts; /* Source dataspace size */
    hid_t src_space_id;             /* Source dataspace ID */

    // Create source dataspace
    if ((src_space_id = H5Screate_simple(1, &dims, NULL)) < 0) {
      fprintf(stderr, "error creating source dataset's dataspace\n");
      goto error;
    }

    // Create the source files and datasets
    for (unsigned u = 0; u < args.ndsets; u++) {
      hid_t src_fid;     /* File ID of source dataset */
      hid_t src_dset_id; /* ID of source dataset */
      char* src_file_name = NULL;

      // Set the file name for the virtual dataset
      asprintf(&src_file_name, "%s_%03u", args.virtual_dset_base_name, u);

      // Create the source HDF5 file
      src_fid = H5Fcreate(src_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
      if (src_fid < 0) {
        fprintf(stderr, "error creating source HDF5 file for virtual datasets\n");
        goto error;
      }

      free(src_file_name);
      src_file_name = NULL;

      // Check whether to fill dataset elements with non-zero value
      if (args.dset_value) {
        // Set dataset fill value 'u + 1', so elements have non-zero values
        if (1 == H5Tequal(args.dset_type, H5T_NATIVE_CHAR)) {
          char c = (char)(u + 1);

          if (H5Pset_fill_value(dcpl_id, H5T_NATIVE_CHAR, &c) < 0) {
            fprintf(stderr, "error setting dataset fill value\n");
            goto error;
          }
        } else if (1 == H5Tequal(args.dset_type, H5T_NATIVE_INT)) {
          int i = (int)(u + 1);

          if (H5Pset_fill_value(dcpl_id, H5T_NATIVE_INT, &i) < 0) {
            fprintf(stderr, "error setting dataset fill value\n");
            goto error;
          }
        } else if (1 == H5Tequal(args.dset_type, H5T_NATIVE_DOUBLE)) {
          double d = (double)(u + 1);

          if (H5Pset_fill_value(dcpl_id, H5T_NATIVE_DOUBLE, &d) < 0) {
            fprintf(stderr, "error setting dataset fill value\n");
            goto error;
          }
        } else {
          fprintf(stderr, "Can't set fill value for unknown datatype\n");
          goto error;
        }
      }

      // Create the source dataset
      src_dset_id = H5Dcreate(
        src_fid, "src_dset", args.dset_type, src_space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
      if (src_dset_id < 0) {
        fprintf(stderr, "error creating source dataset\n");
        goto error;
      }

      // Close the source dataset
      if (H5Dclose(src_dset_id) < 0) {
        fprintf(stderr, "error closing HDF5 dataset\n");
        goto error;
      }

      // Close the source file
      if (H5Fclose(src_fid) < 0) {
        fprintf(stderr, "error closing source HDF5 file\n");
        goto error;
      }
    }

    // Close the source dataspace
    if (H5Sclose(src_space_id) < 0) {
      fprintf(stderr, "error closing source HDF5 dataspace\n");
      goto error;
    }
  }

  // Close the file access property list
  if (H5Pclose(fapl_id) < 0) {
    fprintf(stderr, "error closing HDF5 file access property list\n");
    goto error;
  }

  // Create common parameters for datasets

  // Dataspace
  sid = H5Screate_simple(1, &args.dset_elmts, NULL);
  if (sid < 0) {
    fprintf(stderr, "error creating HDF5 dataspace\n");
    goto error;
  }

  // Create datasets
  for (unsigned u = 0; u < args.ndsets; u++) {
    char* dset_name = NULL;
    hid_t dset_id;

    asprintf(&dset_name, "dset_%03u", u);
    if (args.verbose) {
      printf("Creating dataset '%s'\n", dset_name);
    }

    // Check whether to fill dataset elements with non-zero value
    if (args.dset_value) {
      // Set dataset fill value 'u + 1', so elements have non-zero values
      if (1 == H5Tequal(args.dset_type, H5T_NATIVE_CHAR)) {
        char c = (char)(u + 1);

        if (H5Pset_fill_value(dcpl_id, H5T_NATIVE_CHAR, &c) < 0) {
          fprintf(stderr, "error setting dataset fill value\n");
          goto error;
        }
      } else if (1 == H5Tequal(args.dset_type, H5T_NATIVE_INT)) {
        int i = (int)(u + 1);

        if (H5Pset_fill_value(dcpl_id, H5T_NATIVE_INT, &i) < 0) {
          fprintf(stderr, "error setting dataset fill value\n");
          goto error;
        }
      } else if (1 == H5Tequal(args.dset_type, H5T_NATIVE_DOUBLE)) {
        double d = (double)(u + 1);

        if (H5Pset_fill_value(dcpl_id, H5T_NATIVE_DOUBLE, &d) < 0) {
          fprintf(stderr, "error setting dataset fill value\n");
          goto error;
        }
      } else {
        fprintf(stderr, "Can't set fill value for unknown datatype\n");
        goto error;
      }
    }

    // Check for virtual datasets
    if (args.virtual_dsets) {
      hid_t srcspace      = H5I_INVALID_HID; /* Source dataspace */
      hid_t vspace        = H5I_INVALID_HID; /* Virtual dset dataspaces */
      hsize_t mdims       = args.dset_elmts; /* Dataspace maximum size */
      char* src_file_name = NULL;

      // Set the file name for the virtual dataset
      asprintf(&src_file_name, "%s_%03u", args.virtual_dset_base_name, u);

      // Clear virtual layout in DCPL
      if (H5Pset_layout(dcpl_id, H5D_VIRTUAL) < 0) {
        fprintf(stderr, "error setting layout property\n");
        free(src_file_name);
        goto error;
      }

      // Create virtual dataspace
      if ((vspace = H5Screate_simple(1, &mdims, NULL)) < 0) {
        fprintf(stderr, "error creating virtual dataset's dataspace\n");
        free(src_file_name);
        goto error;
      }

      // Create source dataspace
      if ((srcspace = H5Screate_simple(1, &mdims, NULL)) < 0) {
        fprintf(stderr, "error creating source dataset's dataspace\n");
        free(src_file_name);
        goto error;
      }

#ifdef QAK
      hsize_t start  = 0;                               /* Hyperslab start */
      hsize_t stride = GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE; /* Hyperslab stride */
      hsize_t count  = H5S_UNLIMITED;                   /* Hyperslab count */
      hsize_t block  = GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE; /* Hyperslab block */

      // Select hyperslabs in virtual space
      if (H5Sselect_hyperslab(vspace, H5S_SELECT_SET, &start, &stride, &count, &block) < 0) {
        fprintf(stderr, "error creating selection in virtual dataspace\n");
        free(src_file_name);
        goto error;
      }
#endif

      // Add virtual layout mapping
      if (H5Pset_virtual(dcpl_id, vspace, src_file_name, "src_dset", srcspace) < 0) {
        fprintf(stderr, "error setting virtual dataset property\n");
        free(src_file_name);
        goto error;
      }

      free(src_file_name);
    }

    // Create the dataset
    dset_id = H5Dcreate(fid, dset_name, args.dset_type, sid, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    if (dset_id < 0) {
      fprintf(stderr, "error creating HDF5 dataset\n");
      goto error;
    }
    free(dset_name);
    dset_name = NULL;

    // Write data to datasets - only for virtual datasets, since the
    // fill-value setting does the writing for non-virtual datasets
    if (0 && args.virtual_dsets) {
      hid_t memspace     = H5I_INVALID_HID;                  // Memory dataspace
      hsize_t elmts_left = args.dset_elmts;                  // Elements left to write
      hsize_t dims       = GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE;  // Memory buffer elements
      void* buf          = NULL;                             // Buffer to hold elements

      // Allocate buffer
      buf = malloc(H5Tget_size(args.dset_type) * GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE);
      if (NULL == buf) {
        fprintf(stderr, "error allocating element buffer\n");
        goto error;
      }

      // Fill the buffer
      if (1 == H5Tequal(args.dset_type, H5T_NATIVE_CHAR)) {
        char* cbuf = (char*)buf;
        char c     = (char)(u + 1);

        for (unsigned v = 0; v < GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE; v++) {
          *cbuf++ = c;
        }
      } else if (1 == H5Tequal(args.dset_type, H5T_NATIVE_INT)) {
        int* ibuf = (int*)buf;
        int i     = (int)(u + 1);

        for (unsigned v = 0; v < GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE; v++) {
          *ibuf++ = i;
        }
      } else if (1 == H5Tequal(args.dset_type, H5T_NATIVE_DOUBLE)) {
        double* dbuf = (double*)buf;
        double d     = (double)(u + 1);

        for (unsigned v = 0; v < GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE; v++) {
          *dbuf++ = d;
        }
      } else {
        fprintf(stderr, "Can't fill buffer for unknown datatype\n");
        free(buf);
        goto error;
      }

      // Create memory dataspace
      if ((memspace = H5Screate_simple(1, &dims, NULL)) < 0) {
        fprintf(stderr, "error creating memory dataspace\n");
        free(buf);
        goto error;
      }

      // Write dataset elements
      // (to virtual dataset, which end up in the source dataset)
      while (elmts_left > 0) {
        hsize_t start; /* Hyperslab start */
        hsize_t count; /* Hyperslab count */

        fprintf(stderr, "%u: elmts_left = %llu\n", __LINE__, (unsigned long long)elmts_left);
        // Check if we are performing I/O on a full buffer
        start = (args.dset_elmts - elmts_left);
        fprintf(stderr, "%u: start = %llu\n", __LINE__, (unsigned long long)start);
        if (elmts_left > GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE) {
          // Select hyperslab in dataset
          count = GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE;
          if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, &start, NULL, &count, NULL) < 0) {
            fprintf(stderr, "error creating selection in dataset's dataspace\n");
            free(buf);
            goto error;
          }
        } else {
          // Select hyperslab in dataset
          count = elmts_left;
          if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, &start, NULL, &count, NULL) < 0) {
            fprintf(stderr, "error creating selection in dataset's dataspace\n");
            free(buf);
            goto error;
          }

          // Select hyperslab in memory
          start = 0;
          if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &start, NULL, &count, NULL) < 0) {
            fprintf(stderr, "error creating selection in memory dataspace\n");
            free(buf);
            goto error;
          }
        }

        // Write the data
        if (H5Dwrite(dset_id, args.dset_type, sid, memspace, H5P_DEFAULT, buf) < 0) {
          fprintf(stderr, "error writing to virtual dataset\n");
          free(buf);
          goto error;
        }

        // Decrement # of elements left to write
        elmts_left -= GEN_H5_DEF_VIRT_DSET_BLOCK_SIZE;
      }

      // Close the memory dataspace
      if (H5Sclose(memspace) < 0) {
        fprintf(stderr, "error closing memory dataspace\n");
        free(buf);
        goto error;
      }

      // Free buffer
      free(buf);
    }

    // Close the dataset
    if (H5Dclose(dset_id) < 0) {
      fprintf(stderr, "error closing HDF5 dataset\n");
      goto error;
    }
  }

  // Close the dataset creation property list
  if (H5Pclose(dcpl_id) < 0) {
    fprintf(stderr, "error closing HDF5 dataset creation property list\n");
    goto error;
  }

  // Close the dataspace
  if (H5Sclose(sid) < 0) {
    fprintf(stderr, "error closing HDF5 dataspace\n");
    goto error;
  }

  // Close the HDF5 file
  if (H5Fclose(fid) < 0) {
    fprintf(stderr, "error closing HDF5 file\n");
    goto error;
  }

  user_params_destroy(&args);
  return 0;

error:
  user_params_destroy(&args);
  return 1;
}
