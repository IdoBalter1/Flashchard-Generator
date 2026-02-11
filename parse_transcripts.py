# save as parse_transcripts.py
import re
import sys
import json
try:
    import pyperclip
except Exception:
    pyperclip = None

def clean(s: str) -> str:
    s = re.sub(r'[\d:]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse(text: str) -> dict:
    # split on lines that look like "Lecture 1", "Lecture 2", "Lec 3", etc.
    pat = re.compile(r'(?m)^\s*(?:lecture|lec|transcript)\s*#?\s*(\d+)\b', re.I)
    matches = list(pat.finditer(text))
    if not matches:
        return {1: clean(text)}
    out = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        num = int(m.group(1))
        out[num] = clean(text[start:end])
    return out

def main():
    if len(sys.argv) > 1:
        src_arg = sys.argv[1]
        if src_arg == '--clipboard':
            if not pyperclip:
                print("pyperclip not installed. Install or pass a filename.")
                return
            raw = pyperclip.paste()
        else:
            with open(src_arg, 'r', encoding='utf-8') as f:
                raw = f.read()
    else:
        if pyperclip:
            raw = pyperclip.paste()
        else:
            print("Usage: python parse_transcripts.py raw.txt  OR  python parse_transcripts.py --clipboard")
            return

    transcripts = parse(raw)
    out_file = 'generated_transcripts.py'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('transcripts = {\n')
        for k in sorted(transcripts):
            f.write(f'    {k}: {json.dumps(transcripts[k], ensure_ascii=False)},\n')
        f.write('}\n')
    print("Wrote", out_file)

if __name__ == '__main__':
    main()