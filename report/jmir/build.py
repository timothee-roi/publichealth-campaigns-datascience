#!/usr/bin/env python3
"""Convert report/main.tex into a JMIR-style manuscript (LaTeX fragments for pandoc).

Writes main.tex and appendix.tex fragments to build/ ; pandoc and post-processing are run
by make.sh / post.py.
"""
import re, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'main.tex'
FIG = HERE.parent / 'figures'
OUT = HERE / 'build'
OUT.mkdir(exist_ok=True)

tex = SRC.read_text(encoding='utf8')
tex = re.sub(r'(?m)(?<!\\)%.*$', '', tex)          # strip comments

i_body = tex.index('\\section{Introduction}')
i_bib = tex.index('\\begin{thebibliography}')
i_app = tex.index('\\appendix')
i_end = tex.index('\\end{document}')
body = tex[i_body:i_bib]
app = tex[i_app + len('\\appendix'):i_end]
bibtxt = tex[i_bib:i_app]

# ---------------------------------------------------------------- helpers
def grab(text, idx):
    """text[idx] == '{': return (content, index after the matching brace)."""
    assert text[idx] == '{', text[idx:idx + 20]
    depth = 0
    for j in range(idx, len(text)):
        c = text[j]
        if c == '\\':
            continue
        if c == '{' and text[j - 1] != '\\':
            depth += 1
        elif c == '}' and text[j - 1] != '\\':
            depth -= 1
            if depth == 0:
                return text[idx + 1:j], j + 1
    raise ValueError('unbalanced')

def unwrap_cmd(text, cmd, nargs_before_body):
    """Replace \\cmd{a}{b}...{body} by body (e.g. \\resizebox{w}{h}{body})."""
    out = ''
    pos = 0
    pat = '\\' + cmd
    while True:
        k = text.find(pat, pos)
        if k < 0:
            out += text[pos:]
            return out
        out += text[pos:k]
        j = k + len(pat)
        args = []
        for _ in range(nargs_before_body + 1):
            while text[j] in ' \n':
                j += 1
            a, j = grab(text, j)
            args.append(a)
        out += args[-1]
        pos = j

def americanize(t):
    pairs = [('Optimised', 'Optimized'), ('optimised', 'optimized'), ('characterise', 'characterize'),
             ('coloured', 'colored'), ('minimising', 'minimizing'), ('neighbours', 'neighbors'),
             ('neighbour', 'neighbor'), ('normalisation', 'normalization'), ('normalised', 'normalized'),
             ('normalising', 'normalizing'), ('rumour', 'rumor'), ('standardised', 'standardized'),
             ('summarises', 'summarizes'), ('synchronise', 'synchronize'), ('centred', 'centered'),
             ('favour', 'favor'), ('grey', 'gray'), ('centre', 'center')]
    for a, b in pairs:
        t = t.replace(a, b)
    return t

def pvalues(t):
    t = re.sub(r'\$p\s*(=|<|>|\\le|\\ge)', lambda m: '$P' + m.group(1), t)
    t = t.replace('$p$-values', '$P$ values').replace('$p$-value', '$P$ value')
    t = t.replace('p-values', 'P values').replace('p-value', 'P value')
    return t

# ---------------------------------------------------------------- floats
FLOAT_RE = re.compile(r'\\begin\{(figure|table)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}', re.S)

def extract_floats(text):
    floats = {}
    def repl(m):
        kind, inner = m.group(1), m.group(2)
        labs = re.findall(r'\\label\{([^}]+)\}', inner)
        label = labs[-1] if labs else 'nolabel%d' % (len(floats) + 1)
        k = inner.rindex('\\caption')
        cap, _ = grab(inner, inner.index('{', k))
        floats[label] = dict(kind=kind, inner=inner, caption=cap.strip(), label=label)
        return '\n\n@@FLOAT:%s@@\n\n' % label
    return FLOAT_RE.sub(repl, text), floats

def place_floats(text, floats):
    """Move each float after the paragraph that first mentions it."""
    orig_pos = {}
    stripped = ''
    last = 0
    for m in re.finditer(r'\n*@@FLOAT:([^@]+)@@\n*', text):
        stripped += text[last:m.start()]
        orig_pos[m.group(1)] = len(stripped)
        stripped += '\n\n'
        last = m.end()
    stripped += text[last:]
    inserts = []   # (para_end, mention_pos, label)
    keep = []
    for lab in floats:
        m = re.search(r'\\ref\{%s\}' % re.escape(lab), stripped)
        # a range \ref{a}--\ref{b} mentions every float of the same kind between a and b
        labs_order = list(floats)
        for rm in re.finditer(r'\\ref\{([^}]+)\}--\\ref\{([^}]+)\}', stripped):
            ra, rb = rm.group(1), rm.group(2)
            if ra in labs_order and rb in labs_order:
                ia, ib = labs_order.index(ra), labs_order.index(rb)
                if ia <= labs_order.index(lab) <= ib and floats[lab]['kind'] == floats[ra]['kind']:
                    if m is None or rm.start() < m.start():
                        m = rm
        if m is None:
            keep.append(lab)
            continue
        e = stripped.find('\n\n', m.start())
        e = len(stripped) if e < 0 else e
        inserts.append((e, m.start(), lab))
    for lab in keep:
        p = orig_pos[lab]
        inserts.append((p, 10 ** 9 + p, lab))
    out = stripped
    by_end = {}
    for e, mp, lab in inserts:
        by_end.setdefault(e, []).append((mp, lab))
    for e in sorted(by_end, reverse=True):
        block = ''.join('\n\n@@FLOAT:%s@@' % lab for _, lab in sorted(by_end[e]))
        out = out[:e] + block + out[e:]
    return out

# ---------------------------------------------------------------- build maps
body_t, body_f = extract_floats(body)
app_t, app_f = extract_floats(app)
body_t = place_floats(body_t, body_f)
app_t = place_floats(app_t, app_f)

numbering = {}      # label -> dict(scope, kind, num)
for scope, text in (('main', body_t), ('app', app_t)):
    cnt = {'figure': 0, 'table': 0}
    for m in re.finditer(r'@@FLOAT:([^@]+)@@', text):
        lab = m.group(1)
        f = (body_f if scope == 'main' else app_f)[lab]
        cnt[f['kind']] += 1
        numbering[lab] = dict(scope=scope, kind=f['kind'], num=cnt[f['kind']])

def disp(lab):
    n = numbering[lab]
    return ('A' if n['scope'] == 'app' else '') + str(n['num'])

# appendix parts
app_parts = {}
for k, m in enumerate(re.finditer(r'\\section\{([^}]+)\}\\label\{(app:[^}]+)\}', app_t)):
    app_parts[m.group(2)] = 'ABCDEFGH'[k]

SECTION_NAMES = {
    'sec:data': 'the Data Sources subsection of Methods',
    'sec:windows': 'the Seasonal Windows subsection of Methods',
    'sec:linear': 'the Linear Granger Causality subsection of Methods',
    'sec:calendar': 'the Accounting for the Campaign Calendar subsection of Methods',
    'sec:nonlinear': 'the Nonlinear Granger Causality subsection of Methods',
    'sec:simulation': 'the Checks on Simulated Data subsection of Methods',
    'sec:beyond': 'the Links Beyond the Campaign Calendar subsection of Results',
}

def resolve_refs(t, scope):
    def where(lab):
        n = numbering[lab]
        if scope == 'main':
            return 'in Multimedia Appendix 1' if n['scope'] == 'app' else ''
        return '(main text)' if n['scope'] == 'main' else ''
    kinds = {'Figure': 'figure', 'Table': 'table'}
    # ranges
    def rng(m):
        a, b = m.group(2), m.group(3)
        w = where(a)
        s = '%ss %s-%s' % (m.group(1), disp(a), disp(b))
        return s + (' ' + w if w else '')
    t = re.sub(r'(?:Appendix\s+)?(Figure|Table)s~\\ref\{([^}]+)\}--\\ref\{([^}]+)\}', rng, t)
    def one(m):
        lab = m.group(2)
        w = where(lab)
        return '%s %s' % (m.group(1), disp(lab)) + (' ' + w if w else '')
    t = re.sub(r'(?:Appendix\s+)?(Figure|Table)~\\ref\{([^}]+)\}', one, t)
    def part(m):
        lab = m.group(1)
        if scope == 'main':
            return 'Multimedia Appendix 1, Part %s' % app_parts[lab]
        return 'Part %s' % app_parts[lab]
    t = re.sub(r'Appendix~\\ref\{(app:[^}]+)\}', part, t)
    t = re.sub(r'Section~\\ref\{(sec:[^}]+)\}', lambda m: SECTION_NAMES[m.group(1)], t)
    t = re.sub(r'\\label\{[^}]*\}', '', t)
    assert '\\ref' not in t, re.findall(r'.{40}\\ref.{40}', t)
    return t

# ---------------------------------------------------------------- floats -> latex
def split_caption(cap):
    cap = cap.strip()
    parts = re.split(r'(?<=[.])\s+(?=[A-Z$\\])', cap)
    return parts

def figure_tex(f):
    inner = f['inner']
    imgs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', inner)
    paths = []
    for n in imgs:
        p = FIG / n
        if not p.suffix:
            for ext in ('.png', '.jpg', '.jpeg', '.pdf'):
                if (FIG / (n + ext)).exists():
                    p = FIG / (n + ext)
                    break
        assert p.exists(), n
        paths.append(str(p))
    w = '\\linewidth' if len(paths) == 1 else '0.48\\linewidth'
    gfx = '\\hfill '.join('\\includegraphics[width=%s]{%s}' % (w, p) for p in paths)
    return '\\begin{center}\n%s\n\\end{center}\n' % gfx

def render_float(lab, scope, f):
    d = disp(lab)
    cap = f['caption']
    if f['kind'] == 'figure':
        return '%s\n\\noindent\\textbf{Figure %s.} %s\n' % (figure_tex(f), d, cap)
    inner = f['inner']
    k = inner.index('\\begin{tabular')
    e = list(re.finditer(r'\\end\{tabularx?\}', inner))[-1]
    tab = inner[k:e.end()]
    tab = re.sub(r'\\end\{tabularx\}', r'\\end{tabular}', tab)
    tab = tab.replace('\\begin{tabularx}{\\linewidth}{lXX}', '\\begin{tabular}{lp{6cm}p{6cm}}')
    tab = re.sub(r'\\cmidrule(?:\([^)]*\))?\{[^}]*\}', '', tab)
    def _sup(m):
        return '%s\u00d710\\textsuperscript{%s}' % (m.group(1), m.group(2).replace('-', '\u2212'))
    tab = re.sub(r'\$(-?[0-9.]+)\\times ?10\^\{(-?[0-9]+)\}\$', lambda m: ('\u2212' if m.group(1).startswith('-') else '') + _sup(re.match(r'(.*)', m.group(1).lstrip('-')) and type('M', (), {'group': lambda self, i, _g=(m.group(1).lstrip('-'), m.group(2)): _g[i - 1]})()), tab)
    tab = re.sub(r'\$([+-]?[0-9.]+)\$', lambda m: m.group(1).replace('-', '\u2212'), tab)
    tab = tab.replace('$\\to$', '\u2192')
    parts = split_caption(cap)
    title, notes = parts[0], parts[1:]
    letters = 'abcdefghijklmnopqrstuvwxyz'
    mark = ''
    if notes:
        mark = '\\textsuperscript{%s}' % (letters[0] if len(notes) == 1 else '%s-%s' % (letters[0], letters[len(notes) - 1]))
    out = '\\noindent\\textbf{Table %s.} %s%s\n\n%s\n' % (d, title, mark, tab)
    for n, note in enumerate(notes):
        out += '\n\\noindent\\textsuperscript{%s}%s\n' % (letters[n], note)
    return out

def render_all(text, floats, scope):
    def repl(m):
        lab = m.group(1)
        return '\n\n' + render_float(lab, scope, floats[lab]) + '\n\n'
    return re.sub(r'@@FLOAT:([^@]+)@@', repl, text)

body_t = render_all(body_t, body_f, 'main')
app_t = render_all(app_t, app_f, 'app')

# ---------------------------------------------------------------- citations
bibkeys = re.findall(r'\\bibitem\{([^}]+)\}', bibtxt)
order = []
def cite_scan(t):
    for m in re.finditer(r'\\cite\{([^}]+)\}', t):
        for k in m.group(1).split(','):
            k = k.strip()
            if k not in order:
                order.append(k)
# the placeholder for the code repository is cited in the data-availability paragraph
def fmt_nums(nums):
    nums = sorted(nums)
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append('%d-%d' % (nums[i], nums[j]))
        else:
            out.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return '[' + ','.join(out) + ']'

# repository citation key
REPO = 'roina_repo'
body_cited = body_t
# ensure repo is numbered after all others cited before the availability paragraph: position by first mention
def cite_sub(t):
    def r(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        return fmt_nums([order.index(k) + 1 for k in keys])
    return re.sub(r'\\cite\{([^}]+)\}', r, t)

# ---------------------------------------------------------------- reference strings
REFS = {
 'akaike1974': 'Akaike H. A new look at the statistical model identification. IEEE Trans Automat Contr 1974;19(6):716-723. doi:10.1109/TAC.1974.1100705',
 'ayers2016': 'Ayers JW, Westmaas JL, Leas EC, et al. Leveraging big data to improve health awareness campaigns: a novel evaluation of the Great American Smokeout. JMIR Public Health Surveill 2016;2(1):e16. doi:10.2196/publichealth.5304 PMID:27227151',
 'dickey1979': 'Dickey DA, Fuller WA. Distribution of the estimators for autoregressive time series with a unit root. J Am Stat Assoc 1979;74(366):427-431. doi:10.1080/01621459.1979.10482531',
 'eysenbach2009': 'Eysenbach G. Infodemiology and infoveillance: framework for an emerging set of public health informatics methods to analyze search, communication and publication behavior on the Internet. J Med Internet Res 2009;11(1):e11. doi:10.2196/jmir.1157',
 'glynn2011': 'Glynn RW, Kelly JC, Coffey N, Sweeney KJ, Kerin MJ. The effect of breast cancer awareness month on internet search activity - a comparison with awareness campaigns for lung and prostate cancer. BMC Cancer 2011;11:442. doi:10.1186/1471-2407-11-442',
 'gonzalezpadilla2021': 'González-Padilla DA, España-Navarro R, Subiela JD, et al. Is "Movember" an effective prostate cancer awareness campaign beyond the English language? Insights from Google Trends among Spanish speakers. Soc Int Urol J 2021;2(6):362-369. doi:10.48083/UHSI5324',
 'granger1969': 'Granger CWJ. Investigating causal relations by econometric models and cross-spectral methods. Econometrica 1969;37(3):424-438.',
 'holm1979': 'Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat 1979;6(2):65-70.',
 'kwak2018': 'Kwak H, An J, Salminen J, Jung SG, Jansen BJ. What we read, what we search: media attention and public attention among 193 countries. In: Proceedings of the 2018 World Wide Web Conference. 2018. doi:10.1145/3178876.3186137',
 'kwiatkowski1992': 'Kwiatkowski D, Phillips PCB, Schmidt P, Shin Y. Testing the null hypothesis of stationarity against the alternative of a unit root: how sure are we that economic time series have a unit root? J Econom 1992;54(1-3):159-178. doi:10.1016/0304-4076(92)90104-Y',
 'marinazzo2008': 'Marinazzo D, Pellicoro M, Stramaglia S. Kernel method for nonlinear Granger causality. Phys Rev Lett 2008;100(14):144103. doi:10.1103/PhysRevLett.100.144103',
 'mccombs1972': 'McCombs ME, Shaw DL. The agenda-setting function of mass media. Public Opin Q 1972;36(2):176-187.',
 'murphy2026': 'Murphy F, Benavoli A. Constraint- and score-based nonlinear Granger causality discovery with kernels. arXiv. Preprint posted online on January 14, 2026. doi:10.48550/arXiv.2601.09579',
 'mavragani2019': 'Mavragani A, Ochoa G. Google Trends in infodemiology and infoveillance: methodology framework. JMIR Public Health Surveill 2019;5(2):e13439. doi:10.2196/13439 PMID:31144671',
 'nuti2014': 'Nuti SV, Wayda B, Ranasinghe I, et al. The use of Google Trends in health care research: a systematic review. PLoS One 2014;9(10):e109583. doi:10.1371/journal.pone.0109583',
 'roberts2021': 'Roberts H, et al. Media Cloud: massive open source collection of global news on the open web. Proc Int AAAI Conf Web Soc Media 2021;15(1). doi:10.1609/icwsm.v15i1.18127',
 'theiler1992': 'Theiler J, Eubank S, Longtin A, Galdrikian B, Farmer JD. Testing for nonlinearity in time series: the method of surrogate data. Physica D 1992;58(1-4):77-94.',
 'weeks2010': 'Weeks B, Southwell B. The symbiosis of news coverage and aggregate online search behavior: Obama, rumors, and presidential politics. Mass Commun Soc 2010;13(4):341-360. doi:10.1080/15205430903470532',
 'schreiber2000': 'Schreiber T. Measuring information transfer. Phys Rev Lett 2000;85(2):461-464. doi:10.1103/PhysRevLett.85.461',
 'kraskov2004': 'Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. Phys Rev E 2004;69(6):066138. doi:10.1103/PhysRevE.69.066138',
 'frenzel2007': 'Frenzel S, Pompe B. Partial mutual information for coupling analysis of multivariate time series. Phys Rev Lett 2007;99(20):204101. doi:10.1103/PhysRevLett.99.204101',
 'barnett2009': 'Barnett L, Barrett AB, Seth AK. Granger causality and transfer entropy are equivalent for Gaussian variables. Phys Rev Lett 2009;103(23):238701. doi:10.1103/PhysRevLett.103.238701',
 REPO: 'Roina T. publichealth-campaigns-datascience: code, daily data and result tables. GitHub. URL: https://github.com/timothee-roi/publichealth-campaigns-datascience [accessed 2026-11-27]',
}
assert set(bibkeys) <= set(REFS), set(bibkeys) - set(REFS)

# ---------------------------------------------------------------- headings & structure
HEAD = {
 'Seasonal windows': 'Seasonal Windows', 'Linear Granger causality': 'Linear Granger Causality',
 'Accounting for the campaign calendar': 'Accounting for the Campaign Calendar',
 'Nonlinear Granger causality': 'Nonlinear Granger Causality',
 'Checks on simulated data': 'Checks on Simulated Data',
 'Temporal alignment of media and search peaks': 'Temporal Alignment of Media and Search Peaks',
 'Stationarity': 'Stationarity', 'Robustness (LOYO)': 'Robustness (LOYO)',
 'Links beyond the campaign calendar': 'Links Beyond the Campaign Calendar',
 'Google Trends (GT)': 'Google Trends (GT)', 'MediaCloud (MC)': 'MediaCloud (MC)',
 'Query construction': 'Query Construction', 'Data quality': 'Data Quality',
 'Within-season design': 'Within-Season Design', 'Lag selection': 'Lag Selection',
 'Inference and robustness': 'Inference and Robustness',
 'Uncertainty and impulse responses': 'Uncertainty and Impulse Responses',
 'Season-swap surrogates': 'Season-Swap Surrogates', 'Calendar conditioning': 'Calendar Conditioning',
 'Calendar adjustment of the series': 'Calendar Adjustment of the Series',
 'Kernel Granger causality (KGC)': 'Kernel Granger Causality (KGC)', 'Transfer entropy (TE)': 'Transfer Entropy (TE)',
 'Lag order': 'Lag Order', 'What the calendar explains': 'What the Calendar Explains',
 'Which part of the calendar do the signals depend on?': 'Which Part of the Calendar Do the Signals Depend On?',
 'Nonlinear tests: simulations, lag order and surrogate distributions': 'Nonlinear Tests: Simulations, Lag Order, and Surrogate Distributions',
 'Query construction': 'Query Construction', 'Stationarity tests': 'Stationarity Tests',
 'Sensitivity of the linear tests to the specification': 'Sensitivity of the Linear Tests to the Specification',
 'Linear Granger causality: lag selection and coefficients': 'Linear Granger Causality: Lag Selection and Coefficients',
 'Calendar components of the series': 'Calendar Components of the Series',
 'Calendar adjustments of the nonlinear tests': 'Calendar Adjustments of the Nonlinear Tests',
}
DROP_PARAGRAPH_HEADS = {'Impulse responses', 'Season fixed effects'}

def hd(t):
    t = t.strip().rstrip('.')
    assert t in HEAD, t
    return HEAD[t]

def restructure_paragraph_heads(t):
    def r(m):
        title = m.group(1).strip().rstrip('.')
        if title in DROP_PARAGRAPH_HEADS:
            return ''
        return '\\subsubsection{%s}\n' % hd(title)
    return re.sub(r'\\paragraph\{([^}]*)\}\s*', r, t)

def restructure_sections(t):
    def r(m):
        return '\\subsection{%s}' % hd(m.group(1)) + '\n'
    t = re.sub(r'\\subsection\{([^}]+)\}(?:\\label\{[^}]*\})?', r, t)
    return t

# split body
secs = re.split(r'(?m)^\\section\{([^}]+)\}(?:\\label\{[^}]*\})?\s*$', body_t)
# secs = [pre, title1, text1, title2, text2 ...]
S = {secs[i]: secs[i + 1] for i in range(1, len(secs), 2)}
print(list(S))

# Introduction
intro = S['Introduction']
chunks = re.split(r'\\paragraph\{([^}]*)\}\s*', intro)
intro_first = chunks[0].strip()
IP = {chunks[i].strip().rstrip('.'): chunks[i + 1].strip() for i in range(1, len(chunks), 2)}
assert list(IP) == ['Related work', 'The problem', 'Question and contributions'], list(IP)
intro_tex = ('\\section{Introduction}\n\\subsection{Background}\n%s\n\n%s\n\n\\subsection{Prior Work}\n%s\n\n'
             '\\subsection{Goal of This Study}\n%s\n') % (intro_first, IP['The problem'], IP['Related work'],
                                                         IP['Question and contributions'])

# Methods = Data (as Data Sources) + Methods
data = S['Data']
data = data.replace('\\paragraph{Reproducibility.}', '\\paragraph{Data and Code Availability.}')
HEAD['Data and Code Availability'] = 'Data and Code Availability'
data = re.sub(r'Code, daily data and all result tables are available at \\url\{[^}]*\}\.',
              'Code, daily data and all result tables are available online \\\\cite{%s}.' % REPO, data)
assert REPO in data
methods = S['Methods']
methods_tex = '\\section{Methods}\n\\subsection{Data Sources}\n' + restructure_paragraph_heads(data) + '\n' + \
              restructure_paragraph_heads(restructure_sections(methods))
# the Data float (overview figure) stays in Data Sources

results = S['Results']
results_tex = '\\section{Results}\n' + restructure_paragraph_heads(restructure_sections(results))

# Discussion
disc = S['Discussion']
dch = re.split(r'\\paragraph\{([^}]*)\}\s*', disc)
DP = {dch[i].strip().rstrip('.'): dch[i + 1].strip() for i in range(1, len(dch), 2)}
assert list(DP) == ['Most of the coupling reflects the campaign calendar', 'Movember', 'Predictive, not causal',
                    'Limitations', 'Scope'], list(DP)
comparison = (
    'Earlier studies of awareness campaigns compared a campaign period with a baseline and found that searches rise '
    'during the campaign \\cite{glynn2011,gonzalezpadilla2021,ayers2016}. Searches in our data also peak during the '
    'campaign month for all four campaigns (Table~\\ref{tab:peaklag}). Studies that paired news volume with search volume found a '
    'link in one direction or the other \\cite{weeks2010,kwak2018}. Without accounting for the calendar, our standard '
    'Granger tests also found at least one significant direction for 7 of the 8 campaign\\textendash direction pairs, '
    'but our simulations show that a calendar shared by two series can produce such links when no day-to-day link '
    'exists, and the season-swap tests removed nearly all of them. We did not apply these tests to the data of the '
    'earlier studies, so we cannot say whether their results reflect a shared calendar. The check is relevant to any '
    'Granger analysis of series that follow the same recurring calendar.')
disc_tex = ('\\section{Discussion}\n\\subsection{Principal Results}\n%s\n\n%s\n\n%s\n\n\\subsection{Limitations}\n%s\n\n%s\n\n'
            '\\subsection{Comparison With Prior Work}\n%s\n\n') % (
    DP['Most of the coupling reflects the campaign calendar'], DP['Movember'], DP['Predictive, not causal'],
    DP['Limitations'], DP['Scope'], comparison)
concl = S['Conclusion and future work']
disc_tex += '\\subsection{Conclusions}\n' + concl.strip() + '\n'

main_tex_body = intro_tex + '\n' + methods_tex + '\n' + results_tex + '\n' + disc_tex
# fix a non-sentence start in the first discussion paragraph? (kept as in source)

# ---------------------------------------------------------------- final cleanup of body
def finish(t, scope):
    t = unwrap_cmd(t, 'resizebox', 2)
    t = re.sub(r'\\texttt\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\(small|footnotesize|centering|noindent)\b\s*(?=[^\n]*)', lambda m: '\\noindent ' if m.group(1) == 'noindent' else '', t)
    t = americanize(t)
    t = pvalues(t)
    t = resolve_refs(t, scope)
    t = cite_sub(t)
    t = t.replace('\\\\cite', '\\cite')
    return t

cite_scan(main_tex_body)
main_final = finish(main_tex_body, 'main')

# appendix
app_secs = re.split(r'(?m)^\\section\{([^}]+)\}(?:\\label\{[^}]*\})?\s*$', app_t)
app_out = ''
for i in range(1, len(app_secs), 2):
    title, text = app_secs[i], app_secs[i + 1]
    lab = re.search(r'\\section\{%s\}\\label\{(app:[^}]+)\}' % re.escape(title), app_t).group(1)
    app_out += '\\section{Part %s. %s}\n' % (app_parts[lab], hd(title)) + restructure_paragraph_heads(text) + '\n'
app_final = finish(app_out, 'app')

# references
nums = {k: n + 1 for n, k in enumerate(order)}
(OUT / 'order.json').write_text(json.dumps(order))
for k in (list(order)):
    pass
refs_list = [REFS[k] for k in order]
ref_tex = ''.join('\\noindent\\hspace{0pt}%d.\\ %s\n\n' % (n + 1, r) for n, r in enumerate(refs_list))
ref_tex = ref_tex.replace('&', '\\&').replace('_', '\\_')

(OUT / 'body.tex').write_text(main_final, encoding='utf8')
(OUT / 'appendix.tex').write_text(app_final, encoding='utf8')
(OUT / 'refs.tex').write_text(ref_tex, encoding='utf8')
(OUT / 'numbering.json').write_text(json.dumps(numbering, indent=1))
print('order of references:', order)
print('floats main:', {k: disp(k) for k, v in numbering.items() if v['scope'] == 'main'})
print('floats app:', {k: disp(k) for k, v in numbering.items() if v['scope'] == 'app'})


# ---------------------------------------------------------------- figure files for the JMIR upload form
from PIL import Image
UP = HERE / 'figures_upload'
UP.mkdir(exist_ok=True)
for old in UP.glob('Figure*.png'):
    old.unlink()
for lab, n in numbering.items():
    if n['scope'] != 'main' or n['kind'] != 'figure':
        continue
    imgs = []
    for nm in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', body_f[lab]['inner']):
        p = FIG / nm
        if not p.suffix:
            p = FIG / (nm + '.png')
        imgs.append(Image.open(p).convert('RGB'))
    if len(imgs) == 1:
        im = imgs[0]
    else:
        h = max(i.height for i in imgs)
        imgs = [i.resize((round(i.width * h / i.height), h)) for i in imgs]
        im = Image.new('RGB', (sum(i.width for i in imgs), h), 'white')
        x = 0
        for i in imgs:
            im.paste(i, (x, 0)); x += i.width
    im.save(UP / ('Figure%d.png' % n['num']), dpi=(200, 200))
