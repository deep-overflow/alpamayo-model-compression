"""Convert analysis_v2.tex / method_v2.tex to Markdown with the text unchanged: only markup is
translated (sections, finding boxes, emphasis, equations, the one table, refs, comments)."""
import re
import sys
from pathlib import Path


def convert(tex: str) -> str:
    lines = tex.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("%"):  # keep comments, as HTML comments
            out.append("<!-- " + s[1:].strip() + " -->")
        elif s.startswith("\\section{"):
            out.append("# " + re.match(r"\\section\{(.*)\}", s).group(1))
        elif s.startswith("\\subsection{"):
            out.append("## " + re.match(r"\\subsection\{(.*)\}", s).group(1))
        elif s.startswith("\\label{"):
            out.append("<!-- label: " + re.match(r"\\label\{(.*)\}", s).group(1) + " -->")
        elif s == "\\begin{finding}":
            body = []
            i += 1
            while lines[i].strip() != "\\end{finding}":
                body.append(lines[i].strip())
                i += 1
            out.append("> " + inline(" ".join(body)))
        elif s.startswith("\\input{"):
            out.append("*[Figure: " + re.match(r"\\input\{(.*)\}", s).group(1) + "]*")
        elif s == "\\begin{equation}":
            body = []
            i += 1
            while lines[i].strip() != "\\end{equation}":
                t = lines[i].strip()
                if not t.startswith("\\label"):
                    body.append(t)
                i += 1
            out.append("$$")
            out.extend(body)
            out.append("$$")
        elif s.startswith("\\begin{table}"):
            i = table(lines, i, out)
        else:
            out.append(inline(ln))
        i += 1
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def table(lines, i, out):
    caption, label, rows = "", "", []
    while lines[i].strip() != "\\end{table}":
        s = lines[i].strip()
        if s.startswith("\\caption{"):
            caption = inline(s[len("\\caption{"):-1])
        elif s.startswith("\\label{"):
            label = re.match(r"\\label\{(.*)\}", s).group(1)
        elif "&" in s and s.endswith("\\\\"):
            cells = [c.strip() for c in s[:-2].split("&")]
            rows.append([inline(c) if c else "" for c in cells])
        elif s in ("\\midrule",):
            rows.append("---")
        i += 1
    out.append(f"**Table [{label}].** {caption}")
    out.append("")
    header, body = rows[0], rows[1:]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join("---" for _ in header) + "|")
    for r in body:
        if r == "---":
            continue
        out.append("| " + " | ".join(r) + " |")
    return i


def inline(s: str) -> str:
    # protect math
    maths = []

    def keep(m):
        maths.append(m.group(0))
        return f"\x00{len(maths) - 1}\x00"

    s = re.sub(r"\$[^$]*\$", keep, s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", s)
    s = re.sub(r"~?\\cite\{([^}]*)\}", r" [\1]", s)
    s = re.sub(r"~\\ref\{([^}]*)\}", r" [\1]", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\subref\{([^}]*)\}", r"[\1]", s)
    s = s.replace("Finding~", "Finding ").replace("Figure~", "Figure ").replace("Sec.~", "Sec. ")
    s = s.replace("\\%", "%").replace("\\,", " ").replace("i.e.\\ ", "i.e. ").replace("{,}", ",")
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("\\_", "_")
    s = re.sub(r"\x00(\d+)\x00", lambda m: maths[int(m.group(1))], s)
    return s


for name in sys.argv[1:]:
    src = Path(name)
    md = convert(src.read_text())
    dst = src.with_suffix(".md")
    dst.write_text(md)
    leftovers = sorted(set(re.findall(r"\\[a-zA-Z]+", re.sub(r"\$[^$]*\$|\$\$[\s\S]*?\$\$|<!--.*?-->", "", md))))
    print(f"{dst}: {md.count(chr(10))} lines; LaTeX commands left outside math/comments: {leftovers}")
