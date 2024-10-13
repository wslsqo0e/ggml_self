#include "ggml.h"

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#include <syscall.h>
#include <execinfo.h>

#include <sys/wait.h>

static void ggml_print_backtrace_symbols(void) {
    void * trace[100];
    int nptrs = backtrace(trace, sizeof(trace)/sizeof(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}

static void ggml_print_backtrace(void) {
    char attach[32];
    snprintf(attach, sizeof(attach), "attach %d", getpid());
    int pid = fork();
    if (pid == 0) {
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (char *) NULL);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", attach,
            (char *) NULL);
        exit(EXIT_FAILURE);
    } else {
        int wstatus;
        waitpid(pid, &wstatus, 0);
        if (WIFEXITED(wstatus)) {
            if (WEXITSTATUS(wstatus) == EXIT_FAILURE) {
                // gdb failed, fallback to backtrace_symbols
                ggml_print_backtrace_symbols();
            }
        }
    }
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);

    fprintf(stderr, "%s:%d: ", file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");

    ggml_print_backtrace();
    abort();
}

const char * ggml_status_to_string(enum ggml_status status) {
    switch (status) {
        case GGML_STATUS_ALLOC_FAILED: return "GGML status: error (failed to allocate memory)";
        case GGML_STATUS_FAILED:       return "GGML status: error (operation failed)";
        case GGML_STATUS_SUCCESS:      return "GGML status: success";
        case GGML_STATUS_ABORTED:      return "GGML status: warning (operation aborted)";
    }

    return "GGML status: unknown";
}

struct ggml_context * ggml_init(struct ggml_init_params params) {
}
