import regex as re
import sys




def tokenizer(text):
    """breaks text within '.', '!' or '?' into new lines, putting each new line within '<s>' and '</s>' and removes
    all different kinds of .-/".!? """

    text = re.sub('\s+',' ', text)  #All lines on the same row
    text = re.sub('([0-9\p{L}\"\:\,\- ]+[\.\!\?])',
                  '<s>' + r' \1' + ' </s>\n', text).lower()
    text = re.sub('([.\?\!\,\"\-])', "", text)
    text = re.sub(' +', ' ', text)
    return text


if __name__ == '__main__':
    text = open('Selma.txt').read()
    text2 = """Hejsan mitt namn är "Gustav", vad heter du? Hej! Jag är 34 år gammal och är en "väldigt" glad 
    människa! Är du också - glad? Det hoppas jag.
    - Men käre far."""
    words = tokenizer(text)
    print(words)