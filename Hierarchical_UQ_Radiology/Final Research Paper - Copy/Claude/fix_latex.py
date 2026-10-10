import re

with open('hierarchical_uq_jcst.tex', 'r', encoding='utf-8') as f:
    text = f.read()

def table_replacer(match):
    if 'resizebox' in match.group(0):
        return match.group(0)
    return r'\resizebox{\textwidth}{!}{' + match.group(0) + '}'

text = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', table_replacer, text, flags=re.DOTALL)

with open('hierarchical_uq_jcst.tex', 'w', encoding='utf-8') as f:
    f.write(text)
