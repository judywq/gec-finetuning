from errant.converter import convert_m2_to_text

def main():
    m2_str = """S It 's difficult answer at the question " what are you going to do in the future ? " if the only one who has to know it is in two minds .
A 3 3|||M:VERB:FORM|||to|||REQUIRED|||-NONE-|||0
A 4 5|||U:PREP||||||REQUIRED|||-NONE-|||0"""

    expected = "It 's difficult to answer the question \" what are you going to do in the future ? \" if the only one who has to know it is in two minds ."
    actual = convert_m2_to_text(m2_str)
    print(actual['corrected'] == expected)
    print(actual['corrected'])
    print(expected)


if __name__ == "__main__":
    main()
