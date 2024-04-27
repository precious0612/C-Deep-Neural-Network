/* utils/rand.h
 *
 * This file provides utility functions for generating random numbers, which are
 * essential in various aspects of deep learning, such as weight initialization,
 * dropout regularization, and data augmentation.
 *
 * Key functionalities include:
 *
 * 1. Generating a random floating-point number within a specified range using
 *    a uniform distribution.
 * 2. Generating a random integer within a specified range using a uniform
 *    distribution.
 *
 * These functions ensure consistent and reproducible random number generation,
 * enabling developers to incorporate randomness into their deep learning models
 * while maintaining control over the random seed for debugging and reproducibility
 * purposes.
 *
 * Usage examples:
 *
 * // Generate a random float between 0 and 1
 * float random_value = rand_uniform(0.0f, 1.0f);
 *
 * // Generate a random integer between 1 and 10
 * int random_int = rand_int(1, 10);
 */

#ifndef RAND_H
#define RAND_H

float rand_uniform(float min, float max);
int rand_int(int min, int max);

#endif // /* RAND_H */