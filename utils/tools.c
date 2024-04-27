/* utils/tools.c */

#include <string.h>
#include "tools.h"

int is_empty_string(const char* str) {
    return str == NULL || str[0] == '\0';
}

int not_empty_string(const char* str) {
    return str != NULL && str[0] != '\0';
}
