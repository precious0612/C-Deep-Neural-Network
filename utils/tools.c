/* utils/tools.c */

#include <string.h>
#include <stdio.h>
#include "tools.h"

#define PROGRESS_BAR_LENGTH 50

int is_empty_string(const char* str) {
    return str == NULL || str[0] == '\0';
}

int not_empty_string(const char* str) {
    return str != NULL && str[0] != '\0';
}

void print_progress_bar(float progress) {
    int filled_length = (int)(progress * PROGRESS_BAR_LENGTH);
    int remaining_length = PROGRESS_BAR_LENGTH - filled_length;

    printf("\r[");
    for (int i = 0; i < filled_length; i++) {
        printf("#");
    }
    for (int i = 0; i < remaining_length; i++) {
        printf("-");
    }
    printf("] %3.0f%%", progress * 100);
    fflush(stdout);

    if (progress >= 1.0) {
        printf("\n");
    }
}