"""
The Trie works by storing words in a tree-like structure, where each node
represents a character in the word. Each path from the root to a leaf node
represents a word in the Trie. The Trie supports three main operations:
inserting a word, searching for a word, and checking if any word starts with
a given prefix.
"""


class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_word: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root  # start at root
        for char in word:  # loop through cars
            if char not in node.children:  # if node for char doesn't exist, add
                node.children[char] = TrieNode()  # create node at char place
            node = node.children[char]  # move to that node
        node.is_word = True  # add end of word

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def next_letters(self, word: str) -> list[str]:
        node = self.root
        for char in word:
            if char not in node.children:
                return []
            node = node.children[char]
        return list(node.children.keys())
