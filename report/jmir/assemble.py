from pathlib import Path
B = Path('build')
body = (B/'body.tex').read_text(encoding='utf8')
app = (B/'appendix.tex').read_text(encoding='utf8')
refs = (B/'refs.tex').read_text(encoding='utf8')

ABSTRACT = r'''
\noindent\textbf{Background:} Awareness campaigns such as Octobre Rose, Movember, Mars Bleu, and Dry January recur at the same time every year, so press coverage and online searches rise together on the same dates. Whether they also follow each other from day to day is rarely quantified, and a calendar shared by both series can make one series seem to predict the other in a Granger causality test.

\noindent\textbf{Objective:} This study aimed to test whether press coverage predicts public search interest, or the reverse, during four French health awareness campaigns, beyond the campaign calendar and the weekly cycle that both series share.

\noindent\textbf{Methods:} We analyzed daily Google Trends search volume and MediaCloud counts of articles from French national online news outlets within campaign windows of 43 to 92 days, for 11 seasons (2015-2025) of each campaign. We compared linear Granger causality tests, kernel Granger causality, and transfer entropy with the same tests run on surrogate data in which the cause series of each season was replaced by that of another season (199 surrogates; 999 for the calendar adjustments), with Holm correction across the 8 campaign-direction pairs. Each procedure was also checked on 100 simulated datasets per scenario in which the true answer was known.

\noindent\textbf{Results:} Peaks of searches and coverage coincided for Octobre Rose and Dry January (within 2 days in 9 of 11 seasons each), whereas for Movember coverage peaked a median of 9 days after searches. At least one standard test was significant for 7 of the 8 campaign-direction pairs. In simulations with a shared fixed calendar and no link, standard tests reported links in 39\% to 100\% of the datasets and season-swap tests in 2\% to 7\%; a purely nonlinear link was detected in 100 of 100 datasets by kernel Granger causality and transfer entropy, but in only 27 to 29 of 100 by the linear test. On the campaign data, 3 of 48 season-swap tests were significant after correction (adjusted P=0.040 each). After removal of the average campaign profile, weekly cycle, and season means, 1 of 24 tests was significant (kernel Granger causality, Mars Bleu, searches to coverage, adjusted P=0.040), and it was not reproduced by the linear test (P=0.62) or transfer entropy (P=0.91). For Octobre Rose, the campaign with the strongest standard tests, the real search series explained no more of press coverage than search series from other years (kernel statistic 0.312 vs 0.325 for surrogates). Transfer entropy detected a strong lagged link in 100 of 100 simulated datasets but a weak one in only 10 of 100.

\noindent\textbf{Conclusions:} Most of the apparent coupling between press coverage and search interest during these campaigns reflects the calendar they share. We found no evidence of a day-to-day link beyond the campaign calendar, although weak links cannot be excluded with 11 seasons per campaign. The associations are predictive and do not show a causal effect. Season-swap checks are advisable in Granger analyses of series that share a recurring calendar.

\noindent\textbf{Keywords:} infodemiology; digital health; health awareness campaigns; Google Trends; media coverage; Granger causality; transfer entropy; surrogate data
'''

FRONT = r'''@@ORIGINALPAPER@@

@@TITLE@@

\section{Abstract}
''' + ABSTRACT

BACK = r'''
\subsection{Acknowledgements}
[To be completed by the authors: author contributions, funding information, financial disclosure, role of sponsors, and other acknowledgements. Omit this section if not applicable.]

\subsection{Conflicts of Interest}
[To be completed by the authors: disclose any personal financial interests related to the subject matter; write "None declared" if there are none.]

\subsection{Abbreviations}
\noindent ADF: augmented Dickey-Fuller

\noindent AIC: Akaike information criterion

\noindent GT: Google Trends

\noindent KGC: kernel Granger causality

\noindent KPSS: Kwiatkowski-Phillips-Schmidt-Shin

\noindent LOYO: leave one year out

\noindent MC: MediaCloud

\noindent RSV: relative search volume

\noindent TE: transfer entropy

\section{Multimedia Appendix 1}
Supplementary methods and results (query construction, stationarity tests, sensitivity analyses, additional figures and tables), provided as a separate file.

\section{References}
''' + refs

(B/'manuscript.tex').write_text(FRONT + body + BACK, encoding='utf8')

APP_FRONT = r'''@@TITLE@@

\noindent Multimedia Appendix to: Media Coverage and Online Search Interest During French Health Awareness Campaigns: Granger Causality and Transfer Entropy Beyond the Campaign Calendar. Supplementary methods and results: query construction, stationarity tests, sensitivity analyses, and additional figures and tables. Figure and table numbers that start with the letter A refer to this file; other numbers refer to the main text.

'''
(B/'appendix_full.tex').write_text(APP_FRONT + app, encoding='utf8')
