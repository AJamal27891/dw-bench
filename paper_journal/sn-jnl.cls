%%
%% Springer Nature sn-jnl LaTeX template (simplified for Journal of Big Data)
%% This is a minimal version; for full template, download from:
%% https://www.springernature.com/gp/authors/campaigns/latex-author-support
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2024/01/01 Springer Nature Journal Class (simplified)]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[twoside,11pt]{article}

% Required packages
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}   % Times font
\RequirePackage{geometry}
\geometry{
  a4paper,
  top=2.5cm, bottom=2.5cm,
  left=2.5cm, right=2.5cm,
  headsep=1cm
}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{amsmath}
\RequirePackage{natbib}
\RequirePackage{hyperref}
\RequirePackage{xcolor}
\RequirePackage{fancyhdr}
\RequirePackage{caption}

% Page style
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{Journal of Big Data}}
\fancyhead[LO]{\textit{DW-Bench}}
\renewcommand{\headrulewidth}{0.4pt}

% Title formatting
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries\@title\par}
    \vskip 1em
    {\large\@author\par}
    \vskip 0.5em
    {\normalsize\@date\par}
  \end{center}
  \vskip 1.5em
}

% Abstract environment
\renewenvironment{abstract}{%
  \small
  \begin{center}
    {\bfseries\abstractname\vspace{-.5em}\vspace{0pt}}
  \end{center}
  \quotation
}{\endquotation}

% Keywords
\newcommand{\keywords}[1]{\par\addvspace\baselineskip\noindent\textbf{Keywords:} #1}

\endinput
