from trie_example import Trie

class Homework(Trie):
    def count_words_with_suffix(self, pattern: str) -> int:
        """
        Підраховує кількість слів, що закінчуються заданим шаблоном (pattern).
        Враховує регістр. Виконує обхід Trie без створення списку всіх слів.
        """
        if not isinstance(pattern, str):
            raise TypeError(f"Illegal argument for count_words_with_suffix: pattern = {pattern} має бути рядком (str)")

        def _count_suffix(node, current_word):
            """Рекурсивно обходить Trie, формуючи слова на льоту."""
            count = 0
            if node.value is not None and current_word.endswith(pattern):
                count += 1
            for char, next_node in node.children.items():
                count += _count_suffix(next_node, current_word + char)
            return count

        total = _count_suffix(self.root, "")
        print(f"Суфікс '{pattern}': знайдено {total} слів, що закінчуються на цей шаблон.")
        return total

        
    def has_prefix(self, prefix) -> bool:
        """
        Перевіряє, чи є хоча б одне слово з заданим префіксом (prefix).
        Враховує регістр. Повертає True або False.
        """
        if not isinstance(prefix, str):
            raise TypeError("Аргумент 'prefix' має бути рядком (str).")

        # Отримуємо усі слова, що починаються з даного префікса (метод уже є в базовому Trie)
        words_with_prefix = self.keys_with_prefix(prefix)
        result = len(words_with_prefix) > 0

        if result:
            print(f"Префікс '{prefix}' знайдено у {len(words_with_prefix)} словах.")
        else:
            print(f"Префікс '{prefix}' не знайдено жодного разу.")

        return result


# -----------------------------
# Демонстрація роботи
# -----------------------------
if __name__ == "__main__":
    trie = Homework()
    words = ["apple", "application", "banana", "cat"]
    for i, word in enumerate(words):
        trie.put(word, i)
        print(f"Додано слово '{word}' до Trie (значення {i})")

    print("\n=== Перевірка кількості слів, що закінчуються на шаблон ===")
    assert trie.count_words_with_suffix("e") == 1    # apple
    assert trie.count_words_with_suffix("ion") == 1  # application
    assert trie.count_words_with_suffix("a") == 1    # banana
    assert trie.count_words_with_suffix("at") == 1   # cat

    print("\n=== Перевірка наявності префікса ===")
    assert trie.has_prefix("app") == True    # apple, application
    assert trie.has_prefix("bat") == False
    assert trie.has_prefix("ban") == True    # banana
    assert trie.has_prefix("ca") == True     # cat

    print("\n✅ Усі перевірки пройдено успішно!")