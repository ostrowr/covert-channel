/* Single user-facing command: native sender, bundled IQ decoder. */
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cc_send_main(int argc, char **argv);

static void usage(void) {
    fputs("Usage:\n"
          "  covert-channel encode --text 'Hello over RF'\n"
          "  covert-channel encode --text-file message.txt\n"
          "  covert-channel decode capture.iq\n", stderr);
}

int main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        usage();
        return argc < 2 ? 2 : 0;
    }
    if (!strcmp(argv[1], "encode")) {
        if (argc < 3) { usage(); return 2; }
        fflush(stdout);
        int saved = dup(STDOUT_FILENO);
        int sink = open("/dev/null", O_WRONLY);
        if (saved < 0 || sink < 0 || dup2(sink, STDOUT_FILENO) < 0) {
            perror("redirect sender output");
            if (saved >= 0) close(saved);
            if (sink >= 0) close(sink);
            return 1;
        }
        close(sink);
        int rc = cc_send_main(argc - 1, argv + 1);
        fflush(stdout);
        if (dup2(saved, STDOUT_FILENO) < 0) { perror("restore output"); rc = 1; }
        close(saved);
        if (rc == 0) puts("Transmission complete.");
        return rc;
    }
    if (!strcmp(argv[1], "decode")) {
        if (argc < 3) { usage(); return 2; }
        char path[PATH_MAX];
        uint32_t size = sizeof(path);
        if (_NSGetExecutablePath(path, &size) != 0) {
            fputs("Could not locate CLI executable\n", stderr);
            return 1;
        }
        char *slash = strrchr(path, '/');
        if (!slash) { fputs("Invalid CLI path\n", stderr); return 1; }
        const char *decoder_name = "covert-channel-decoder";
        if ((size_t)(path + sizeof(path) - (slash + 1)) <= strlen(decoder_name)) {
            fputs("CLI path is too long\n", stderr);
            return 1;
        }
        strcpy(slash + 1, decoder_name);
        char **child = calloc((size_t)argc, sizeof(*child));
        if (!child) { perror("calloc"); return 1; }
        child[0] = path;
        for (int i = 2; i < argc; ++i) child[i - 1] = argv[i];
        child[argc - 1] = NULL;
        execv(path, child);
        fprintf(stderr, "Could not start decoder at %s: %s\n", path, strerror(errno));
        free(child);
        return 1;
    }
    usage();
    return 2;
}
