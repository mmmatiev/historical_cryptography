import math
import numpy as np


class SubstitutionCiphers:
    class SimpleReplacementShifter:
        @staticmethod
        def simple_replacement_shifter_encode(message, alphabet, key):
            """
            Кодирует сообщение с использованием простого шифра замены с заданным алфавитом и ключом.

            Args:
                message (str): Сообщение, которое нужно закодировать.
                alphabet (str): Алфавит, используемый для кодирования. Каждому символу в сообщении
                               должен соответствовать символ в этом алфавите.
                key (str): Ключ для шифра. Он должен иметь такую же длину, как и алфавит,
                            и каждому символу в алфавите должен соответствовать символ
                            в ключе для замены.

            Returns:
                str: Закодированное сообщение.

            Raises:
                ValueError: Если длина алфавита и ключа не совпадает.
            """
            encrypted_text = ''
            if len(list(alphabet)) == len(list(key)):
                for i in range(len(message)):
                    if message[i] not in alphabet:
                        encrypted_text += message[i]
                    else:
                        encrypted_text += list(key)[list(alphabet).index(message[i])]
                return encrypted_text
            else:
                raise ValueError('некорректный ввод алфавита и/или ключа: их размер должен быть одинаковым')

        @staticmethod
        def simple_replacement_shifter_decode(message, alphabet, key):
            """
                Декодирует сообщение, используя простой шифр замены с заданным алфавитом и ключом.

                Args:
                    message (str): Закодированное сообщение, которое нужно декодировать.
                    alphabet (str): Алфавит, используемый при кодировании. Каждому символу в
                                    закодированном сообщении должен соответствовать символ в этом алфавите.
                    key (str): Ключ для шифра. Он должен иметь такую же длину, как и алфавит,
                               и каждому символу в ключе должен соответствовать символ в алфавите
                               для обратной замены.

                Returns:
                    str: Декодированное сообщение.

                Raises:
                    ValueError: Если длина алфавита и ключа не совпадает.
                """
            decrypted_text = ''
            if len(list(alphabet)) == len(list(key)):
                for i in range(len(message)):
                    if message[i] not in alphabet:
                        decrypted_text += message[i]
                    else:
                        decrypted_text += list(alphabet)[list(key).index(message[i])]
                return decrypted_text
            else:
                raise ValueError('некорректный ввод алфавита и/или ключа: их размер должен быть одинаковым')

    class AffineShifter:
        @staticmethod
        def affine_shifter_encode(message, alphabet, key):
            """
            Шифрует сообщение с использованием аффинного шифра с заданным алфавитом и ключом.

            Args:
                message (str): Сообщение, которое нужно зашифровать.
                alphabet (str): Алфавит, используемый для шифрования. Каждому символу в
                                сообщении должен соответствовать символ в этом алфавите.
                key (tuple): Ключ для аффинного шифра, представляющий собой кортеж из двух целых чисел (a, b).
                             Число a должно быть взаимно простым с длиной алфавита. Число b обычно является сдвигом.

            Returns:
                str: Зашифрованное сообщение.

            Raises:
                ValueError: Если НОД(a, длина алфавита) не равен 1.
            """
            encrypted_text = ''
            a, b = key[0], key[1]
            if math.gcd(a, len(list(alphabet))) == 1:
                for i in range(len(message)):
                    if message[i] not in alphabet:
                        encrypted_text += message[i]
                    else:
                        encrypted_text += alphabet[
                            (list(alphabet).index(message[i]) * a + b) % int(len(list(alphabet)))]
                return encrypted_text
            else:
                raise ValueError('НОД(a) и мощность алфавита должны быть равны 1')

        @staticmethod
        def affine_shifter_decode(message, alphabet, key):
            """
            Дешифрует сообщение, используя аффинный шифр с заданным алфавитом и ключом.

            Args:
                message (str): Зашифрованное сообщение, которое нужно декодировать.
                alphabet (str): Алфавит, используемый при дешифровании. Каждому символу в
                                зашифрованном сообщении должен соответствовать символ в этом алфавите.
                key (tuple): Ключ для аффинного шифра, представляющий собой кортеж из двух целых чисел (a, b).
                             Число a должно быть взаимно простым с длиной алфавита. Число b обычно является сдвигом.

            Returns:
                str: Дешифрованное сообщение.

            Raises:
                ValueError: Если НОД(a, мощность алфавита) не равен 1.
            """
            decrypted_text = ''
            a, b = key[0], key[1]
            if math.gcd(a, len(list(alphabet))) == 1:
                for i in range(len(message)):
                    if message[i] not in alphabet:
                        decrypted_text += message[i]
                    else:
                        time_variable = (list(alphabet).index(message[i]) - b)
                        while int(time_variable / a) != (time_variable / a):
                            # здесь у нас нормальная арифметика остатков для нецелых чисел
                            # (например чтобы (8/5) % 5 мы должны 8 дополнить до 8+26+26 = 60,
                            # чтобы оно нацело делилось).
                            time_variable += len(list(alphabet))

                        decrypted_text += alphabet[int(time_variable / a) % int(len(list(alphabet)))]
                return decrypted_text
            else:
                raise ValueError('НОД(a, мощность алфавита) должен быть равен 1')

        @staticmethod
        def affine_recurrent_shifter_encode(message, alphabet, key1, key2):
            """
                Шифрует сообщение с использованием аффинного рекуррентного шифра с заданным алфавитом и ключами.

                Args:
                    message (str): Сообщение, которое нужно зашифровать.
                    alphabet (str): Алфавит, используемый для шифрования. Каждому символу в
                                    сообщении должен соответствовать символ в этом алфавите.
                    key1 (tuple): Первый ключ для аффинного рекуррентного шифра, представляющий собой кортеж
                                  из двух целых чисел (a1, b1). Число a1 должно быть взаимно простым с длиной алфавита.
                                  Число b1 обычно является сдвигом.
                    key2 (tuple): Второй ключ для аффинного рекуррентного шифра, представляющий собой кортеж
                                  из двух целых чисел (a2, b2). Число a2 должно быть взаимно простым с длиной алфавита.
                                  Число b2 обычно является сдвигом.

                Returns:
                    str: Зашифрованное сообщение.

                Raises:
                    ValueError: Если НОД(a1, мощность алфавита) или НОД(a2, мощность алфавита) не равны 1.

                """
            encrypted_text = ''
            key_i = [key1, key2]
            if math.gcd(key1[0], len(list(alphabet))) == 1 and math.gcd(key2[0], len(list(alphabet))) == 1:
                for i in range(len(message) - 2):  # 2 ключа есть, а значит нужно вычислить len(message) - 2 ключей
                    key_i += [((key1[0] * key2[0]) % (len(alphabet)), (key1[1] + key2[1]) % (len(alphabet)))]
                    key1, key2 = key2, [(key1[0] * key2[0]) % (len(alphabet)), (key1[1] + key2[1]) % (len(alphabet))]
                for i in range(len(message)):
                    if message[i] not in alphabet:
                        encrypted_text += message[i]
                    else:
                        encrypted_text += alphabet[
                            (list(alphabet).index(message[i]) * key_i[i][0] + key_i[i][1]) % int(len(list(alphabet)))]
                return encrypted_text
            else:
                raise ValueError('НОД(a1, мощность алфавита) и НОД(a2, мощность алфавита) должны быть равны 1')

        @staticmethod
        def affine_recurrent_shifter_decode(message, alphabet, key1, key2):
            """
            Дешифрует сообщение, используя аффинный рекуррентный шифр с заданным алфавитом и ключами.

            Args:
                message (str): Зашифрованное сообщение, которое нужно декодировать.
                alphabet (str): Алфавит, используемый при дешифровании. Каждому символу в
                                зашифрованном сообщении должен соответствовать символ в этом алфавите.
                key1 (tuple): Первый ключ для аффинного рекуррентного шифра, представляющий собой кортеж
                              из двух целых чисел (a1, b1). Число a1 должно быть взаимно простым с длиной алфавита.
                              Число b1 обычно является сдвигом.
                key2 (tuple): Второй ключ для аффинного рекуррентного шифра, представляющий собой кортеж
                              из двух целых чисел (a2, b2). Число a2 должно быть взаимно простым с длиной алфавита.
                              Число b2 обычно является сдвигом.

            Returns:
                str: Дешифрованное сообщение.

            Raises:
                ValueError: Если НОД(a1, мощность алфавита) или НОД(a2, мощность алфавита) не равны 1.
            """
            if math.gcd(key1[0], len(list(alphabet))) == 1 and math.gcd(key2[0], len(list(alphabet))) == 1:
                decrypted_text = ''
                key_i = [key1, key2]
                for i in range(len(message) - 2):
                    key_i += [((key1[0] * key2[0]) % (len(alphabet)), (key1[1] + key2[1]) % (len(alphabet)))]
                    key1, key2 = key2, [(key1[0] * key2[0]) % (len(alphabet)), (key1[1] + key2[1]) % (len(alphabet))]
                for i in range(len(message)):
                    if message[i] not in alphabet:
                        decrypted_text += message[i]
                    else:
                        time_variable = (list(alphabet).index(message[i]) - key_i[i][1])
                        while int(time_variable / key_i[i][0]) != (time_variable / key_i[i][0]):
                            # здесь у нас нормальная арифметика остатков для нецелых чисел
                            # (например чтобы (8/5) % 5 мы должны 8 дополнить до 8+26+26 = 60,
                            # чтобы оно нацело делилось).
                            time_variable += len(list(alphabet))
                        decrypted_text += alphabet[int(time_variable / key_i[i][0]) % int(len(list(alphabet)))]
                return decrypted_text
            else:
                raise ValueError('НОД(a1, мощность алфавита) и НОД(a2, мощность алфавита) должны быть равны 1')


class BlockCipher:
    class OperationsForMatricesModulo:
        @staticmethod
        def matrix_to_array(matrix):
            """
            Преобразует матрицу в двумерный массив.

            Args:
                matrix (list of lists): Матрица, которую нужно преобразовать в массив.

            Returns:
                list of lists: Двумерный массив, представляющий собой матрицу.
            """
            array = []
            for row in matrix:
                array.append(list(row))
            return array

        @staticmethod
        def matrix_minor(matrix, i, j):
            """
            Возвращает минор матрицы, удаляя i-тую строку и j-тый столбец.

            Args:
                matrix (list of lists): Матрица, из которой нужно получить минор.
                i (int): Индекс строки, которую нужно удалить.
                j (int): Индекс столбца, который нужно удалить.

            Returns:
                list of lists: Минор матрицы.
            """
            return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]

        @staticmethod
        def matrix_determinant(matrix):
            """
            Вычисляет определитель матрицы.

            Args:
                matrix (list of lists): Матрица, определитель которой нужно вычислить.

            Returns:
                int or float: Определитель матрицы.
            """
            if len(matrix) == 2:
                return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

            determinant = 0
            for c in range(len(matrix)):
                determinant += ((-1) ** c) * matrix[0][c] * BlockCipher.OperationsForMatricesModulo.matrix_determinant(
                    BlockCipher.OperationsForMatricesModulo.matrix_minor(matrix, 0, c))
            return determinant

        @staticmethod
        def matrix_inverse(m, alphabet):
            """
                Вычисляет обратную матрицу для заданной матрицы в алфавитном пространстве.

                Args:
                    m (list of lists): Матрица, для которой нужно вычислить обратную матрицу.
                    alphabet (str): Алфавит, используемый при вычислении обратной матрицы.

                Returns:
                    list of lists: Обратная матрица.

                Raises:
                    ValueError: Если определитель матрицы равен нулю.
            """
            determinant = BlockCipher.OperationsForMatricesModulo.matrix_determinant(m)
            if len(m) == 2:
                time_variable = [[m[1][1], -1 * m[0][1]],
                                 [-1 * m[1][0], m[0][0]]]
                for i in range(len(time_variable)):
                    for j in range(len(time_variable[0])):
                        matrix_cell = time_variable[i][j]
                        while int(matrix_cell / determinant) != (matrix_cell / determinant):
                            matrix_cell += len(alphabet)
                        time_variable[i][j] = int((matrix_cell / determinant))
                        while time_variable[i][j] < 0:
                            time_variable[i][j] += len(alphabet)
                return time_variable
            cofactors = []
            for r in range(len(m)):
                cofactor_row = []
                for c in range(len(m)):
                    minor = BlockCipher.OperationsForMatricesModulo.matrix_minor(m, r, c)
                    time_variable = ((-1) ** (r + c)) * BlockCipher.OperationsForMatricesModulo.matrix_determinant(
                        minor)
                    while int(time_variable / determinant) != (time_variable / determinant):
                        # здесь у нас нормальная арифметика остатков для нецелых чисел
                        # (например чтобы (8/5) % 5 мы должны 8 дополнить до 8+26+26 = 60,
                        # чтобы оно нацело делилось.
                        time_variable += len(alphabet)
                    cofactor_row.append(time_variable)
                cofactors.append(cofactor_row)
            cofactors = np.transpose(cofactors)
            for r in range(len(cofactors)):
                for c in range(len(cofactors)):
                    cofactors[r][c] = cofactors[r][c] / determinant
                    while cofactors[r][c] < 0:
                        cofactors[r][c] += len(alphabet)
            return BlockCipher.OperationsForMatricesModulo.matrix_to_array(cofactors)

    class HillCipher:
        @staticmethod
        def split_to_chunk(input_string, chunk_size):
            """
            Делит входную строку на куски указанного размера и, при необходимости, дополняет последний кусок
            незначащими символами.

            Args:
                input_string (str): Входная строка, которую нужно разделить на куски.
                chunk_size (int): Размер каждого куска.

            Returns:
                list of lists: Список списков символов. Каждый внутренний список содержит символы одного куска.

            Notes:
                - Если длина входной строки не делится нацело на размер куска,
                  последний кусок будет дополнен символами, введенными пользователем.
            """
            chunks = [input_string[i:i + chunk_size] for i in range(0, len(input_string), chunk_size)]
            padding_needed = len(chunks[-1]) < chunk_size
            if padding_needed:
                padding_length = chunk_size - len(chunks[-1])
                chunks[-1] += str(input('Введите незначащий символ: ')) * padding_length
            chunks = [list(chunk) for chunk in chunks]
            return chunks

        @staticmethod
        def Hill_cipher_encode(message, alphabet, key):
            """
            Шифрует сообщение с использованием шифра Хилла.

            Args:
                message (str): Сообщение, которое нужно зашифровать.
                alphabet (str): Алфавит, используемый при шифровании.
                key (list of lists): Ключ шифрования в виде матрицы.

            Returns:
                str: Зашифрованное сообщение.

            Raises:
                ValueError: Если НОД детерминанта ключа и мощности алфавита не равен 1.
            """
            if math.gcd(BlockCipher.OperationsForMatricesModulo.matrix_determinant(key), len(list(alphabet))) == 1:
                encrypted_text = []
                message = BlockCipher.HillCipher.split_to_chunk(message, len(key))
                for i in range(len(message)):
                    message_j = [alphabet.index(message[i]) for message[i] in message[i]]
                    encrypted = BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                        ([np.dot(np.array(key), np.transpose(message_j)) % len(alphabet)]))
                    encrypted_text += [alphabet[i] for i in encrypted[0]]
                return ''.join(encrypted_text)
            else:
                raise ValueError('НОД детерминанта ключа и мощность алфавита должны быть равны 1')

        @staticmethod
        def Hill_cipher_decode(message, alphabet, key):
            """
            Дешифрует сообщение, зашифрованное с использованием шифра Хилла.

            Args:
                message (str): Зашифрованное сообщение, которое нужно декодировать.
                alphabet (str): Алфавит, используемый при декодировании.
                key (list of lists): Ключ дешифрования в виде матрицы.

            Returns:
                str: Дешифрованное сообщение.

            Raises:
                ValueError: Если НОД детерминанта ключа и мощности алфавита не равен 1.
            """
            if math.gcd(BlockCipher.OperationsForMatricesModulo.matrix_determinant(key), len(list(alphabet))) == 1:
                encrypted_text = []
                message = BlockCipher.HillCipher.split_to_chunk(message, len(key))
                key = BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                    BlockCipher.OperationsForMatricesModulo.matrix_inverse(key, alphabet))
                for i in range(len(message)):
                    message_j = [alphabet.index(message[i]) for message[i] in message[i]]
                    encrypted = BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                        ([np.dot(np.array(key), np.transpose(message_j)) % len(alphabet)]))
                    encrypted_text += [alphabet[i] for i in encrypted[0]]
                return ''.join(encrypted_text)
            else:
                raise ValueError('НОД детерминанта ключа и мощность алфавита должны быть равны 1')

        @staticmethod
        def recurrent_Hill_cipher_encode(message, alphabet, key1, key2):
            """
            Шифрует сообщение с использованием рекуррентного шифра Хилла.

            Args:
                message (str): Сообщение, которое нужно зашифровать.
                alphabet (str): Алфавит, используемый при шифровании.
                key1 (list of lists): Первый ключ шифрования в виде матрицы.
                key2 (list of lists): Второй ключ шифрования в виде матрицы.

            Returns:
                str: Зашифрованное сообщение.

            Raises:
                ValueError: Если НОД детерминантов ключей и мощности алфавита не равен 1.
            """
            if math.gcd(BlockCipher.OperationsForMatricesModulo.matrix_determinant(key1),
                        len(list(alphabet))) == 1 and math.gcd(
                BlockCipher.OperationsForMatricesModulo.matrix_determinant(key2), len(list(alphabet))) == 1:
                encrypted_text = []
                key_i = [key1, key2]
                for i in range(len(message) - 2):
                    key_i += (BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                        ((BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                            np.dot(BlockCipher.OperationsForMatricesModulo.matrix_to_array(np.array(key_i[-2])),
                                   BlockCipher.OperationsForMatricesModulo.matrix_to_array(np.array(key_i[-1]))) % len(
                                alphabet)),))))
                message = BlockCipher.HillCipher.split_to_chunk(message, len(key_i[1]))
                for i in range(len(message)):
                    message_j = [alphabet.index(message[i]) for message[i] in message[i]]
                    encrypted = BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                        ([np.dot(np.array(key_i[i]), np.transpose(message_j)) % len(alphabet)]))
                    encrypted_text += [alphabet[i] for i in encrypted[-1]]
                return ''.join(encrypted_text)
            else:
                raise ValueError('НОД детерминанта ключей и мощность алфавита должны быть равны 1')

        @staticmethod
        def recurrent_Hill_cipher_decode(message, alphabet, key1, key2):
            """
            Дешифрует сообщение, зашифрованное с использованием рекуррентного шифра Хилла.

            Args:
                message (str): Зашифрованное сообщение, которое нужно декодировать.
                alphabet (str): Алфавит, используемый при декодировании.
                key1 (list of lists): Первый ключ дешифрования в виде матрицы.
                key2 (list of lists): Второй ключ дешифрования в виде матрицы.

            Returns:
                str: Дешифрованное сообщение.

            Raises:
                ValueError: Если НОД детерминантов ключей и мощности алфавита не равен 1.
            """
            if math.gcd(BlockCipher.OperationsForMatricesModulo.matrix_determinant(key1),
                        len(list(alphabet))) == 1 and math.gcd(
                BlockCipher.OperationsForMatricesModulo.matrix_determinant(key2),
                len(list(alphabet))) == 1:
                key_i = [key1, key2]
                for i in range(len(message) - 2):
                    key_i += (BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                        ([BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                            np.dot(BlockCipher.OperationsForMatricesModulo.matrix_to_array(np.array(key_i[-2])),
                                   BlockCipher.OperationsForMatricesModulo.matrix_to_array(np.array(key_i[-1]))) % len(
                                alphabet))])))
                encrypted_text = []
                message = BlockCipher.HillCipher.split_to_chunk(message, len(key_i[-1]))
                for i in range(len(message)):
                    key = BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                        BlockCipher.OperationsForMatricesModulo.matrix_inverse(key_i[i], alphabet))
                    message_j = [(alphabet).index(message[i]) for message[i] in message[i]]
                    encrypted = BlockCipher.OperationsForMatricesModulo.matrix_to_array(
                        ([np.dot(np.array(key), np.transpose(message_j)) % len(alphabet)]))
                    encrypted_text += [alphabet[i] for i in encrypted[0]]
                return ''.join(encrypted_text)
            else:
                raise ValueError('НОД детерминанта ключей и мощность алфавита должны быть равны 1')


class GammaCiphers:
    class VigenerCipherWithRepeatKey:
        @staticmethod
        def repeat_key(message, alphabet, key):
            """
            Генерирует повторяющийся гамму на основе ключа длины сообщения.

            Args:
                message (str): Сообщение, для которого требуется сгенерировать последовательность гаммы.
                alphabet (str): Набор символов, используемых для кодирования.
                key (str): Ключ, который нужно повторить для соответствия длине сообщения.

            Returns:
                list: Список целых чисел, представляющий собой последовательность гаммы, сгенерированную из ключа.

            """
            gamma = key * (len(message) // len(key)) + key[:len(message) % len(key)]
            gamma = list(gamma)
            for i in range(len(gamma)):
                gamma[i] = alphabet.index(gamma[i])
            return gamma

        @staticmethod
        def Vigener_cipher_with_repeat_key_encode(message, alphabet, key):
            """
            Кодирует сообщение с использованием шифра Виженера с повторяющимся ключом.

            Args:
                message(str): Сообщение, которое требуется закодировать.
                alphabet (str): Алфавит, используемый в шифре.
                key: (str): Ключ, который будет повторен для кодирования.


            Returns:
                str: Закодированное сообщение.
            """
            gamma = GammaCiphers.VigenerCipherWithRepeatKey.repeat_key(message, alphabet, key)
            encrypted_text = []
            message = list(message)
            for i in range(len(message)):
                if message[i] not in alphabet:
                    encrypted_text += message[i]
                else:
                    message[i] = alphabet.index(message[i])
                    encrypted_text += [str((message[i] + gamma[i]) % len(list(alphabet)))]
                    encrypted_text[i] = list(alphabet)[int(encrypted_text[i])]
            return ''.join(encrypted_text)

        @staticmethod
        def Vigener_cipher_with_repeat_key_decode(message, alphabet, key):
            """
            Расшифровывает сообщение с использованием шифра Виженера с повторяющимся ключом.

            Args:
                message(str): Сообщение, которое требуется расшифровать.
                alphabet (str): Алфавит, используемый в шифре.
                key: (str): Ключ, который будет повторен для кодирования.

            Returns:
                str: Дешифрованное сообщение.
            """
            gamma = GammaCiphers.VigenerCipherWithRepeatKey.repeat_key(message, alphabet, key)
            decrypted_text = []
            message = list(message)
            for i in range(len(gamma)):
                if message[i] not in alphabet:
                    decrypted_text += message[i]
                else:
                    message[i] = alphabet.index(message[i])
                    decrypted_text += [str((message[i] - gamma[i]))]
                    if int(decrypted_text[i]) < 0:
                        decrypted_text[i] = int(decrypted_text[i])
                        decrypted_text[i] += len(list(alphabet))
                        decrypted_text[i] = str(decrypted_text[i])
                    decrypted_text[i] = list(alphabet)[int(decrypted_text[i])]
            return ''.join(decrypted_text)

    class VigenerCipherWithOpenTextKey:
        @staticmethod
        def Vigener_cipher_with_open_text_key_encode(message, alphabet, key):
            """
            Кодирует сообщение с использованием шифра Виженера с самоключом по открытому тексту.

            Args:
                message(str): Сообщение, которое требуется закодировать.
                alphabet (str): Алфавит, используемый в шифре.
                key: (str): Ключ, который будет 1 символом в гамме.

            Returns:
                str: Зашифрованное сообщение.
            """
            key = alphabet.index(key)
            gamma = [alphabet[key]]
            gamma += message[:(len(str(message)) - 1)]
            gamma = list(gamma)
            for i in range(len(gamma)):
                gamma[i] = alphabet.index(gamma[i])
            encrypted_text = []
            message = list(message)
            for i in range(len(message)):
                message[i] = alphabet.index(message[i])
                encrypted_text += [str((message[i] + gamma[i]) % len(list(alphabet)))]
                encrypted_text[i] = list(alphabet)[int(encrypted_text[i])]
            return ''.join(encrypted_text)

        @staticmethod
        def Vigener_cipher_with_open_text_key_decode(message, alphabet, key):
            """
            Расшифровывает сообщение с использованием шифра Виженера с самоключом по открытому тексту.

            Args:
                message(str): Сообщение, которое требуется расшифровать.
                alphabet (str): Алфавит, используемый в шифре.
                key: (str): Ключ, который будет 1 символом в гамме.

            Returns:
                str: Дешифрованное сообщение.
            """
            key = alphabet.index(key)
            gamma = [key]
            decrypted_text = []
            message = list(message)
            for i in range(len(message)):
                message[i] = alphabet.index(message[i])
                decrypted_text += [((message[i] - int(gamma[-1])) % len(alphabet))]
                gamma.append(str(alphabet.index(alphabet[decrypted_text[-1]])))
            return ''.join(alphabet[num] for num in decrypted_text)

    class VigenerCipherWithCipherTextKey:
        @staticmethod
        def Vigener_cipher_with_cipher_text_key_encode(message, alphabet, key):
            """
            Кодирует сообщение с использованием шифра Виженера с самоключом по зашифрованному тексту.

            Args:
                message(str): Сообщение, которое требуется закодировать.
                alphabet (str): Алфавит, используемый в шифре.
                key: (str): Ключ, который будет 1 символом в гамме.

            Returns:
                str: Зашифрованное сообщение.
            """
            key = alphabet.index(key)
            gamma = [key]
            encrypted_text = []
            message = list(message)
            for i in range(len(message)):
                message[i] = alphabet.index(message[i])
                encrypted_text += [((message[i] + int(gamma[-1])) % len(alphabet))]
                gamma.append(str(alphabet.index(alphabet[encrypted_text[-1]])))
            return ''.join(alphabet[num] for num in encrypted_text)

        @staticmethod
        def Vigener_cipher_with_cipher_text_key_decode(message, alphabet, key):
            """
            Расшифровывает сообщение с использованием шифра Виженера с самоключом по зашифрованному тексту.

            Args:
                message(str): Сообщение, которое требуется расшифровать.
                alphabet (str): Алфавит, используемый в шифре.
                key: (str): Ключ, который будет 1 символом в гамме.

            Returns:
                str: Дешифрованное сообщение.
            """
            key = alphabet.index(key)
            gamma = [alphabet[key]]
            gamma += message[:(len(str(message)) - 1)]
            gamma = list(gamma)
            for i in range(len(gamma)):
                gamma[i] = alphabet.index(gamma[i])
            decrypted_text = []
            message = list(message)
            for i in range(len(message)):
                message[i] = alphabet.index(message[i])
                decrypted_text += [str((message[i] - gamma[i]) % len(list(alphabet)))]
                decrypted_text[i] = list(alphabet)[int(decrypted_text[i])]
            return ''.join(decrypted_text)
