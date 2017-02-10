/*
 * mnist.cc
 */

#include <getopt.h>
#include <stdio.h>
#include <time.h>

#include "util.h"      /* timestamp etc. */
#include "mem.h"       /* memory management for matrices */
#include "mat.h"       /* matrix operations */
#include "data.h"      /* read data and draw samples */
#include "functions.h" /* functions in the network (softmax, etc.) */
#include "score.h"     /* record classification and cross entropy errors */

/* the instance of a memory manager */
mem_manager mm;

/* parameters you probably never need to change */
enum {
  n_classes = 16,
  pixel_height = 28,
  pixel_width = 28,
  n_pixels = pixel_height * pixel_width,
  n_training_data = 60000,
  n_test_data = 10000,
};

/* parameters you might want to change to optimize
   learning perofmance and optimize speed.
   you can change them by giving -Dn_units=X
   and/or -Dbatch_sz=X to the command line
   (in Makefile) */

/* number of units in the hidden layer */
#ifndef n_units
#define n_units 128
#endif

/* the maximum number of samples in a single
   mini batch (you can adjust the actual number
   of samples at runtime without changing this value) */
#ifndef max_batch_sz
#define max_batch_sz 1000
#endif

/* parameters set by command line options */
struct mnist_opt {
  /* total number of samples in the training phase */
  long samples_to_train;
  /* total number of samples in the test phase */
  long samples_to_test;
  /* learning rate */
  double eta;
  /* batch size */
  long batch_sz;
  /* control how often it shows progress
     (recent classificatin score and cross entropy error) */
  long progress_interval;
  /* the number of last samples used for scoring
     classificationn/cross-entropy error */
  long average_window_sz;
  /* seed of the random number generator used to initialize weight matrices
     (initial weight matrices are deterministic given the same seed) */
  unsigned long weight_seed;
  /* seed of the random number generator used to draw random samples in the
     training
     (drawn samples are deterministic given the same seed) */
  unsigned long draw_seed;
  /* prefix of filenames to write samples to */
  const char* sample_image_prefix;
  /* max number of sample image files */
  long max_sample_images;
  /* the factor by which images are magnified (1 for no magnification) */
  long magnify_images;

  mnist_opt() {
    /* default values of command line options */
    samples_to_train = 60000;
    samples_to_test = 10000;
    eta = 0.005;
    batch_sz = 128;
    weight_seed = 729918723710L;
    draw_seed = 314159265358L;
    progress_interval = 1000;
    average_window_sz = 10000;
    sample_image_prefix = "imgs/sample";
    max_sample_images = 0;
    magnify_images = 1;
  }
};

void usage(const char* prog) {
  mnist_opt opt;
  fprintf(stderr,
          "usage:\n"
          "  %s [options]\n",
          prog);
  fprintf(stderr,
          "options:\n"
          "  -n,--train N\n"
          "    train with N samples (%ld)\n"
          "  -t,--test N\n"
          "    test with N samples (%ld)\n"
          "  -e,--eta x\n"
          "    set initial learning rate eta (%f)\n"
          "  -b,--batch-sz x\n"
          "    set mini-batch size (%ld)\n"
          "  -p,--progress-interval N\n"
          "    if positive, show classification performance/error every N "
          "samples (%ld) (set to 0 when measuring performance)\n"
          "  -a,--average-window-sz N\n"
          "    the number of last samples to show classification/cross entropy "
          "errors (%ld)\n"
          "  -w,--weight-seed X\n"
          "    random seed for weight matrices (%ld)\n"
          "  -d,--draw-seed X\n"
          "    random seed for drawing samples (%ld)\n"
          "  -S,--sample-image-prefix PREFIX\n"
          "    dump samples into image files <PREFIX>000000_<c>.ppm, "
          "<PREFIX>000001_<c>.ppm, ... (%s) (do not use when measuring "
          "performance)\n"
          "  -N,--max-sample-images N\n"
          "    dump up to N images (%ld)\n"
          "  -g,--magnify-images N (%ld)\n"
          "    magnify dumped images N times\n",
          opt.samples_to_train, opt.samples_to_test, opt.eta, opt.batch_sz,
          opt.progress_interval, opt.average_window_sz, opt.weight_seed,
          opt.draw_seed, opt.sample_image_prefix, opt.max_sample_images,
          opt.magnify_images);
}

mnist_opt parse_opt(int argc, char** argv) {
  mnist_opt opt;
  static struct option long_options[] = {
      {"train", required_argument, 0, 'n'},
      {"test", required_argument, 0, 't'},
      {"eta", required_argument, 0, 'e'},
      {"batch-sz", required_argument, 0, 'b'},
      {"weight-seed", required_argument, 0, 'w'},
      {"draw-seed", required_argument, 0, 'd'},
      {"progress-interval", required_argument, 0, 'p'},
      {"average-window-sz", required_argument, 0, 'a'},
      {"sample-image-prefix", required_argument, 0, 'S'},
      {"max-sample-images", required_argument, 0, 'N'},
      {"magnify-images", required_argument, 0, 'g'},
  };
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "n:t:e:b:w:d:p:a:S:N:g:", long_options,
                        &option_index);
    if (c == -1) break;
    switch (c) {
      case 0:
        printf("option %s", long_options[option_index].name);
        if (optarg) printf(" with arg %s", optarg);
        printf("\n");
        break;
      case 'n':
        opt.samples_to_train = atol(optarg);
        break;
      case 't':
        opt.samples_to_test = atol(optarg);
        break;
      case 'e':
        opt.eta = atof(optarg);
        break;
      case 'b':
        opt.batch_sz = atol(optarg);
        break;
      case 'w':
        opt.weight_seed = atol(optarg);
        break;
      case 'd':
        opt.draw_seed = atol(optarg);
        break;
      case 'p':
        opt.progress_interval = atol(optarg);
        break;
      case 'a':
        opt.average_window_sz = atol(optarg);
        break;
      case 'S':
        opt.sample_image_prefix = strdup(optarg);
        break;
      case 'N':
        opt.max_sample_images = atol(optarg);
        break;
      case 'g':
        opt.magnify_images = atol(optarg);
        break;
      default:
        usage(argv[0]);
        exit(1);
        break;
    }
  }
  return opt;
}

double get_time() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec + ts->tv_nsec * 1.0e-9;
}

/* the procedure for training a neural network
   of three layers, for hand-written digit recognition
   (MNIST dataset).

   W0, W1 and W2 give the initial values of
   the three weight matrices and are updated during
   training.

   the computation represented by this network is essentially:

   x2 = ReLU(W0  x);
   x4 = ReLU(W1 x2);
    e = cross_entropy(softmax(W2 x4), t);

    where x is a vector representing an image of 28x28=784
    gray pixels (i.e., x has 784 elements, each in a floating point number).

    see the slide for the definition of ReLU, softmax, and
    cross entropy.

    the network tries to minimize the expected value of e by
    modifying W0, W1 and W2.  to this end, it views e as a
    function of W0, W1 and W2, and changes each of them
    to the direction of gradients of e with respect to
    W0, W1 and W2, respectively.

    see the slide for how it exactly works.

 */
double train_mnist(mat<n_pixels, n_units> W0, mat<n_units, n_units> W1,
                   mat<n_units, n_classes> W2, mnist_opt opt) {
  /* files to read training data (and their classes) from */
  const char* train_x =
      "data/train_images_60000x784_float32.npy"; /* training data */
  const char* train_y =
      "data/train_labels_60000_float32.npy"; /* their classes */
  mat<n_training_data, n_pixels> X =
      map_npy_file<n_training_data, n_pixels>(train_x);

  mat<n_training_data, 1> C = map_npy_file<n_training_data, 1>(train_y);
  /* X(i,:) is the ith training data, a vector of n_pixels elements.
     C(i,0) is the class of the ith training data (0 - 9).
     all elements are real (which is defined as float) */

  /* total number of samples to train */
  long n_samples_to_train = opt.samples_to_train;
  /* random number generator to draw samples randomly */
  unsigned short rg[3] = {(unsigned short)((opt.draw_seed >> 32) & 65535),
                          (unsigned short)((opt.draw_seed >> 16) & 65535),
                          (unsigned short)((opt.draw_seed >> 0) & 65535)};
  /* lerning rate; fixed to 0.01 for simplicity; you might want to
     make it adaptive to get a better result */
  real eta = opt.eta;
  /* the number of most recent samples it calculates the
     classification scores with. for example, if it is set to 10000,
     it periodically reports classification scores of the last 10000
     samples */
  score_counter sc(opt.average_window_sz);
  /* controls how frequently scores are reported.
     the number of samples trained between two consecutive reports */
  const long progress_interval = opt.progress_interval;
  /* buffer to generate a filename to dump a sample into */
  char filename[strlen(opt.sample_image_prefix) + 100];

  /* the main loop */
  double t0 = get_time();
  tsc_t c0 = get_tsc();

  long bs = opt.batch_sz;
  for (long i = 0; i < n_samples_to_train; i += bs) {
    /* draw a number of samples (max_batch_sz samples except for the last
       iteration,
       which draws whatever number is necessary to end up with exactly the
       specified
       number of samples in total);
       get indices of drawn samples into samples array */
    long ns =
        (i + bs <= n_samples_to_train ? (long)bs : n_samples_to_train - i);
    long samples[ns];
    choose_random_samples<n_training_data, max_batch_sz>(rg, samples, ns);

    /* make a matrix of mini-batch (rows of samples and their classes) by
       actually copying
       rows from X and C */
    mat<max_batch_sz, n_pixels> x =
        get_rows<max_batch_sz, n_pixels, n_training_data>(X, samples, ns);
    mat<max_batch_sz, 1> c =
        get_rows<max_batch_sz, 1, n_training_data>(C, samples, ns);

    /* if requested, dump samples into image files, up to the specified number
       (by --max-sample-images) of samples */
    for (long d = 0; d < ns; d++) {
      if (i + d >= opt.max_sample_images) break;
      sprintf(filename, "%s%06ld_%d.ppm", opt.sample_image_prefix, i + d,
              (int)c(d, 0));
      dump_as_ppm(&x(d, 0), pixel_height, pixel_width, opt.magnify_images,
                  filename);
    }

    /* forward computation, which computes the output of the neural network
       for chosen vectors */
    /* the first layer */
    mat<max_batch_sz, n_units> x1 =
        x * W0;  // <ns,n_pixels> * <n_pixels,n_units>
    mat<max_batch_sz, n_units> x2 = relu2(x1, x1);
    /* the second (hidden) layer */
    mat<max_batch_sz, n_units> x3 =
        x2 * W1;  // <ns,n_units>  * <n_units,n_units>
    mat<max_batch_sz, n_units> x4 = relu2(x3, x3);
    /* the last (output) layer */
    mat<max_batch_sz, n_classes> x5 =
        x4 * W2;                         // <ns,n_units>  * <n_units,n_classes>
    mat<max_batch_sz, 1> p = argmax(x5); /* the predicted class */
    mat<max_batch_sz, 1> e =
        softmax_cross_entropy(x5, c); /* error we try to decrease */

    /* backward computation, which computes gradients of e with respect to W0,
       W1 and W2
       g_xxx represents gradient of e with respect to xxx. see the slide for
       what it's exactly doing and how it works */
    mat<max_batch_sz, n_classes> g_x5 = softmax_minus_one(x5, c);
    mat<max_batch_sz, n_units> g_x4 =
        g_x5 * W2.T();  // <ns,n_classes> * <n_classes,n_units>
    mat<max_batch_sz, n_units> g_x3 = relu2(g_x4, x3);
    mat<max_batch_sz, n_units> g_x2 =
        g_x3 * W1.T();  // <ns,n_units> * <n_units,n_units>
    mat<max_batch_sz, n_units> g_x1 = relu2(g_x2, x1);
    mat<n_units, n_classes> g_W2 =
        x4.T() * g_x5;  // <n_units,ns> * <ns,n_classes>
    mat<n_units, n_units> g_W1 = x2.T() * g_x3;  // <n_units,ns> * <ns,n_units>
    mat<n_pixels, n_units> g_W0 = x.T() * g_x1;  // <n_pixels,ns> * <ns,n_units>

    /* now we got gradients with respect to W0, W1 and W2 (g_W0, g_W1 and g_W2).
       update them with respective gradients */
    W0 -= eta * g_W0;
    W1 -= eta * g_W1;
    W2 -= eta * g_W2;

    /* record the result we got in this iteration */
    sc.update_score(p, c, e);
    /* and report performance (classification errors and cross entropy error)
       with the specified frequency */
    long i_next = i + ns;
    if (opt.progress_interval > 0 &&
        i_next / progress_interval > i / progress_interval) {
      tsc_t dc = diff_tsc(get_tsc(), c0);
      double dt = get_time() - t0;

      printf(
          "training %ld - %ld at %lld clocks and %.8f sec :"
          " classification %ld / %ld = %f cross entropy error = %.8f\n",
          (i_next - sc.n > 0 ? i_next - sc.n : 0), i_next, dc, dt, sc.sum_c,
          sc.n, sc.sum_c / (double)sc.n, sc.sum_e / sc.n);
    }
    /* free all intermediate matrices we allocated, except the ones we use
       across iterations */
    mm.free_except5(W0.a, W1.a, W2.a, X.a, C.a);
  }
  sc.fini();
  /* return the number of flops we performed, counting only those in
     matrix-matrix
     mulplications (ignoring others) */
  return 2.0 * (2.0 * n_pixels * n_units + 3.0 * n_units * n_units +
                3.0 * n_units * n_classes) *
         n_samples_to_train;
}

/* test phase.
   test phase does almost the same computation as the training phase,
   with the only important difference being that the test phase
   does not perform backward computation.

   the same commens are omitted (or made terser). see train_mnist for
   more detailed comments.
 */
double test_mnist(mat<n_pixels, n_units> W0, mat<n_units, n_units> W1,
                  mat<n_units, n_classes> W2, mnist_opt opt) {
  /* files */
  const char* test_x = "data/test_images_10000x784_float32.npy";
  const char* test_y = "data/test_labels_10000_float32.npy";
  mat<n_test_data, n_pixels> X = map_npy_file<n_test_data, n_pixels>(test_x);
  mat<n_test_data, 1> C = map_npy_file<n_test_data, 1>(test_y);
  /* number of samples to test */
  long n_samples_to_test = opt.samples_to_test;

  /* track performance */
  score_counter sc(n_samples_to_test);
  long bs = opt.batch_sz;
  double t0 = get_time();
  tsc_t c0 = get_tsc();
  for (long i = 0; i < n_samples_to_test; i += bs) {
    /* get samples (unlike training phase, we simply get them sequentially) */
    long ns = (i + bs <= n_samples_to_test ? (long)bs : n_samples_to_test - i);
    long samples[ns];
    get_seq_samples<n_test_data, max_batch_sz>(i, samples, ns);
    /* form matrices of test data and their classes */
    mat<max_batch_sz, n_pixels> x =
        get_rows<max_batch_sz, n_pixels, n_test_data>(X, samples, ns);
    mat<max_batch_sz, 1> c =
        get_rows<max_batch_sz, 1, n_test_data>(C, samples, ns);

    /* forward computation; exactly the same as that in training phase */
    mat<max_batch_sz, n_units> x1 =
        x * W0;  // <ns,n_pixels> * <n_pixels,n_units>
    mat<max_batch_sz, n_units> x2 = relu2(x1, x1);
    mat<max_batch_sz, n_units> x3 =
        x2 * W1;  // <ns,n_units>  * <n_units,n_units>
    mat<max_batch_sz, n_units> x4 = relu2(x3, x3);
    mat<max_batch_sz, n_classes> x5 =
        x4 * W2;  // <ns,n_units>  * <n_units,n_classes>
    mat<max_batch_sz, 1> e = softmax_cross_entropy(x5, c);
    mat<max_batch_sz, 1> y = argmax(x5);

    /* record the performance */
    sc.update_score(y, c, e);
    mm.free_except5(W0.a, W1.a, W2.a, X.a, C.a);
  }
  /* report the performance of all test data */
  tsc_t dc = diff_tsc(get_tsc(), c0);
  double dt = get_time() - t0;
  printf(
      "test %ld - %ld at %lld clocks and %.8f sec :"
      " classification %ld / %ld = %f cross entropy error = %.8f\n",
      (n_samples_to_test - sc.n > 0 ? n_samples_to_test - sc.n : 0),
      (long)n_samples_to_test, dc, dt, sc.sum_c, sc.n, sc.sum_c / (double)sc.n,
      sc.sum_e / sc.n);
  sc.fini();
  /* return the number of flops we performed */
  return 2.0 * (1.0 * n_pixels * n_units + 1.0 * n_units * n_units +
                1.0 * n_units * n_classes) *
         n_samples_to_test;
}

/* the main function */
int main(int argc, char** argv) {
  /* parse command line options (set various options) */
  mnist_opt opt = parse_opt(argc, argv);
  /* generate the initial matrices */
  unsigned short rg[3] = {(unsigned short)((opt.weight_seed >> 32) & 65535),
                          (unsigned short)((opt.weight_seed >> 16) & 65535),
                          (unsigned short)((opt.weight_seed >> 0) & 65535)};

  /* initialize the memory manager that allocates memory for
     matrix operations. max_alloc_per_iter determines
     the number of allocated blocks that can be alive at a time.
     you probably do not have to change this; see mem.h for details */
  const long max_alloc_per_iter = 128;
  mm.init(max_alloc_per_iter);

  /* the three matrices in the neural network */
  mat<n_pixels, n_units> W0(n_pixels);
  mat<n_units, n_units> W1(n_units);
  mat<n_units, n_classes> W2(n_units);
  /* initialize their elements with normal distributions */
  init_normal(rg, W0);
  init_normal(rg, W1);
  init_normal(rg, W2);

  printf("samples for training: %ld\n", opt.samples_to_train);
  printf("samples for testing: %ld\n", opt.samples_to_test);
  printf("mini batch size: %ld\n", opt.batch_sz);
  printf("learning rate: %f\n", opt.eta);
  printf("max mini batch size: %d\n", max_batch_sz);
  printf("hidden units: %d\n", n_units);
  printf("pixels per image: %d\n", n_pixels);
  printf("output classes: %d\n", n_classes);

  printf("seed for weight: %ld\n", opt.weight_seed);
  printf("seed for drawing samples: %ld\n", opt.draw_seed);
  printf("interval between reports: %ld\n", opt.progress_interval);
  printf("average window size: %ld\n", opt.average_window_sz);
  printf("sample images written: %ld\n", opt.max_sample_images);
  if (opt.max_sample_images > 0) {
    printf("  with prefix: %s\n", opt.sample_image_prefix);
  }
  printf("images are magnified by a factor: %ld\n", opt.magnify_images);
  printf("language: %s\n", "C++");
  printf("library: %s\n", "none");

  /* ------------- do the main job (train and test) ------------- */
  double t0 = get_time();
  tsc_t c0 = get_tsc();
  double flops_train = train_mnist(W0, W1, W2, opt);
  double t1 = get_time();
  tsc_t c1 = get_tsc();
  double flops_test = test_mnist(W0, W1, W2, opt);
  double t2 = get_time();
  tsc_t c2 = get_tsc();

  /* report FLOPS */
  tsc_t dc_train = diff_tsc(c1, c0);
  tsc_t dc_test = diff_tsc(c2, c1);
  double dt_train = t1 - t0;
  double dt_test = t2 - t1;
  printf("%.0f flops in %lld clocks / %.8f sec to train (%.2f flops/clock)\n",
         flops_train, dc_train, dt_train, flops_train / dc_train);
  printf("%.0f flops in %lld clocks / %.8f sec to test (%.2f flops/clock)\n",
         flops_test, dc_test, dt_test, flops_test / dc_test);

#if 0
  /* if requested, write hidden units we ended up with */
  char filename[strlen(opt.unit_image_prefix) + 100];
  for (long d = 0; d < opt.max_unit_images; d++) {
    if (d >= n_units) break;
    sprintf(filename, "%s%04ld.ppm", opt.unit_image_prefix, d);
    dump_as_ppm(&W0(d, 0), pixel_height, pixel_width, opt.magnify_images, filename);
  }
#endif

  /* free all matrices (W0, W1 and W2) */
  mm.free_all();
  mm.fini();
  return 0;
}
