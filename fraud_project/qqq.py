def caesar_shift(char, shift):
    if char.isupper():
        start = ord('A')
        return chr((ord(char) - start + shift) % 26 + start)
    elif char.islower():
        start = ord('a')
        return chr((ord(char) - start + shift) % 26 + start)
    else:
        return char


text = input()
words = text.split(' ')
result = []

for word in words:
    # Подсчитываем количество букв в слове для сдвига
    letter_count = 0
    for ch in word:
        if ch.isalpha():
            letter_count += 1

    # Шифруем слово
    encrypted_word = ''
    for ch in word:
        encrypted_word += caesar_shift(ch, letter_count)

    result.append(encrypted_word)

print(' '.join(result))