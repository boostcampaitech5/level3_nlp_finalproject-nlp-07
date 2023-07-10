from hanspell import spell_checker
import pandas as pd


def spell_check(text):
    result = spell_checker.check(text)
    return result.checked


if __name__ == "__main__":
    file_name = 'product_list_20230707_152355'

    df = pd.read_csv(f'../{file_name}.csv')
    df['text'] = df['text'].apply(lambda x: spell_check(x))
    df.to_csv(f'../hanspell_{file_name}.csv', index=False)
