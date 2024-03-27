# Историческая криптография на Python

Этот проект представляет собой реализацию различных методов исторической криптографии на языке программирования Python. В нем содержатся классы и функции для работы с классическими шифрами замены, блочным шифрованием, а также шифрами гаммирования.


## SubstitutionCiphers
Класс `SimpleReplacementShifter` содержит классы `SimpleReplacementShifter` и `AffineShifter`.


### SimpleReplacementShifter
Класс `SimpleReplacementShifter` представляет собой реализацию шифра простой замены.

- `simple_replacement_shifter_encode(text)`: Метод для шифрования текста с помощью шифра простой замены.
- `simple_replacement_shifter_decode(text)`: Метод для дешифрования текста, зашифрованного с использованием шифра простой замены.

### AffineShifter
Класс `AffineShifter` реализует аффинный шифр.

- `affine_shifter_encode(text)`: Метод для шифрования текста с помощью  аффинного шифра.
- `affine_shifter_decode(text)`: Метод для дешифрования текстас помощью  аффинного шифра.


## BlockCipher
Класс `BlockCipher` содержит классы `OperationsForMatricesModulo` и `HillCipher`.


### OperationsForMatricesModulo
Класс `OperationsForMatricesModulo` содержит функции для работы с матрицами по модулю.

- `matrix_to_array(matrix)`: Функция для преобразования матрицы в массив.
- `matrix_minor(matrix, i, j)`: Функция для вычисления минора матрицы.
- `matrix_determinant(matrix)`: Функция для вычисления определителя матрицы.
- `matrix_inverse(matrix)`: Функция для вычисления обратной матрицы.

### HillCipher
Класс `HillCipher` реализует блочный шифр Хилла.

- `split_to_chunk(text, chunk_size)`: Функция для разделения текста на блоки определенного размера.
- `Hill_cipher_encode(text)`: Метод для шифрования текста с помощью шифра Хилла.
- `Hill_cipher_decode(text)`: Метод для дешифрования текста, зашифрованного шифром Хилла.


## GammaCiphers
Класс `GammaCiphers` содержит классы `VigenerCipherWithRepeatKey`, `VigenerCipherWithOpenTextKey`, `VigenerCipherWithCipherTextKey`.


### VigenerCipherWithRepeatKey
Класс `VigenerCipherWithRepeatKey` реализует шифр Виженера с повторяющимся ключом.

### VigenerCipherWithOpenTextKey
Класс `VigenerCipherWithOpenTextKey` реализует шифр Виженера с открытым текстом в качестве ключа.

### VigenerCipherWithCipherTextKey
Класс `VigenerCipherWithCipherTextKey` реализует шифр Виженера с шифртекстом в качестве ключа.

## Дополнительная информация

Для получения более подробной информации о каждом классе и функции обратитесь к комментариям в исходном коде.
