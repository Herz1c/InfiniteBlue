# Random sparse positions (two kings + up to 4 other pieces) for testing game-end rules,
# especially insufficient material. Prints one ICN per line.
#   python3 tools/endgame_positions.py 3000 > data/m1/endgames.icn
import random
import sys

BORDER = "-9223372036854773806,9223372036854773815,-9223372036854773806,9223372036854773815"


def main(n, seed=1):
    rng = random.Random(seed)
    for _ in range(n):
        squares = set()

        def square():
            while True:
                s = (rng.randint(-6, 13), rng.randint(-3, 12))
                if s not in squares:
                    squares.add(s)
                    return s

        pieces = []
        for king in ("K", "k"):
            x, y = square()
            pieces.append(f"{king}{x},{y}")
        for _ in range(rng.randint(0, 4)):
            c = rng.choice("QRBNPBBNqrbnpbbn")
            x, y = square()
            if c in "Pp":
                y = rng.randint(2, 7)  # pawns stay off their promotion ranks
                while (x, y) in squares:
                    x += 1
                squares.add((x, y))
            pieces.append(f"{c}{x},{y}")
        turn = rng.choice("wb")
        print(f"{turn} 0/100 1 (8|1) {BORDER} {'|'.join(pieces)}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
