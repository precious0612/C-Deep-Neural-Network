/* utils/tools.h
 *
 * This file provides a set of utility functions for performing common string
 * operations, which are frequently used throughout the codebase. These functions
 * help ensure consistent and efficient handling of string data, reducing the risk
 * of errors and simplifying the development process.
 *
 * Key functionalities include:
 *
 * 1. Checking if a given string is empty or not.
 * 2. Checking if a given string is not empty.
 *
 * By providing these simple yet essential string operations, this header file
 * serves as a centralized location for common string-related utility functions,
 * promoting code reusability and maintainability.
 *
 * Usage examples:
 *
 * // Check if a string is empty
 * const char* empty_str = "";
 * if (is_empty_string(empty_str)) {
 *     // Handle empty string
 * }
 *
 * // Check if a string is not empty
 * const char* non_empty_str = "Hello, World!";
 * if (not_empty_string(non_empty_str)) {
 *     // Process non-empty string
 * }
 */

#ifndef TOOLS_H
#define TOOLS_H

int is_empty_string(const char* str);
int not_empty_string(const char* str);

#endif /* TOOLS_H */