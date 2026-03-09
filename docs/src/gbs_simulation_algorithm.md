# Mathematical background

In this page we will analyse the algorithm for simulating the outcomes of a
Gaussian boson-sampling experiment in more detail.
Since the formulae are already quite heavy by themselves, we will not use the
braket notation (except for the outer product \\(\outp{u}{v}\\) of two vectors
\\(u\\) and \\(v\\)), opting for a lighter mathematical style.

## Gaussian states

We start by laying out some notation and fundamental facts about Gaussian states
that we will need later.

### Definition

Our context is the Hilbert space of \\(n\\) bosonic modes.  For a single mode
the space is the Fock space \\(\fockb(\C) \iso \lseq(\C)\\), and for \\(n\\)
modes we have \\(\fockb(\C)\tsp{n} \iso \fockb(\C^n)\\).  For \\(m\in\N\\), we
call \\(\ns{m}\in\\fockb(\C)\\) the eigenvector of the number operator with
eigenvalue \\(m\\), and for a multi-index
\\(\multi{m}=(m\sb1, \dotsc, m\sb{n})\in\N^n\\) we define
\\(\ns{\multi{m}} \in \fockb(\C)\tsp{n}\\) as \\(\ns{m\sb1} \otimes \dotsb
\otimes \ns{m\sb{n}}\\).
These vectors form an orthonormal basis of the overall Fock space; \\(\ns{0}\\)
is also the vacuum vector, which we will denote by \\(\vacuum\\).

Let \\(R\\) be a vector collecting the single-mode position and momentum
operators as \\(\xpvec\sb{2j-1} \defeq x\sb{j}\\) and \\(\xpvec\sb{2j} \defeq
p\sb{j}\\), for \\(j\in\\{1,\dotsc,n\\}\\).  A Gaussian state \\(\rho\\) on
\\(\fockb(\C^n)\\) is completely determined by its first and second moments,
gathered in the vector \\(r\\) and in the _covariance matrix_ \\(\covmat\\)
defined as

```math
\begin{gather*}
    \fmom_j \defeq
    \tr(\rho \xpvec_j),\\
    \sigma_{jk} \defeq
    \tr\bigl(\rho \fcomm{(\xpvec-\fmom)_j, (\xpvec-\fmom)_k}\bigr) =
    \tr(\rho \fcomm{\xpvec_j,\xpvec_k}) - 2\fmom_j\fmom_k
\end{gather*}
```

for \\(j,k\in\\{1,\dotsc,2n\\}\\).
Covariance matrices are always positive-definite and satisfy
\\(\covmat+\iu\sympmat \geq 0\\), where

```math
\sympmat \defeq \imat[n] \otimes
\begin{pmatrix}
  0  & 1\\
  -1 & 0
\end{pmatrix}
```

is the reference symplectic matrix, such that \\([\xpvec\sb{j}, \xpvec\sb{k}] =
\iu\sympmat\sb{jk}\\).
We will denote by \\(\gauss{\fmom,\covmat}\\) the Gaussian state given by its
covariance matrix \\(\covmat\\) and the vector \\(\fmom\\) of its first moments,
or just \\(\gauss{\covmat}\\) if \\(\fmom=0\\), as is often the case.

### The symplectic group

The symplectic group \\(\Sp{2n,\R}\\) is the group \\(\set{ S \in \Mat{2n,\R} }{
S \sympmat \transpose{S} = \sympmat }\\) where \\(\sympmat\\) is a
skew-symmetric \\(2n \times 2n\\) real matrix such that \\(\sympmat^2 =
-\imat[2n]\\).  Every symplectic matrix has determinant \\(\pm 1\\), is
invertible and \\(S^{-1} = -\sympmat \transpose{S} \sympmat\\).

!!! info "Alternative notation for symplectic matrices"
    There are other choices for the skew-symmetric matrix \\(\sympmat\\) that
    defines the properties of the symplectic group: another popular choice is
   
    ```math
    \sympmat \defeq
    \begin{pmatrix}
      0  & 1 \\ -1 & 0
    \end{pmatrix}
    \otimes
    \imat[n],
    ```

    with which some properties can be written in a clearer way.
    We can switch back and forth between these two notations with a permutation
    matrix \\(Y\\) such that \\(\imat[n] \otimes \begin{psmallmatrix} 0 & 1 \\\\
    -1 & 0 \end{psmallmatrix} = Y \bigl(\begin{psmallmatrix} 0 & 1 \\\\ -1 & 0
    \end{psmallmatrix} \otimes \imat[n]\bigr) \transpose{Y}\\).

Let \\(\UtoSp \colon \Mat{n,\C} \to \Mat{2n,\R}\\) be the map

```math
\UtoSp(M) =
Y \begin{pmatrix}
    M\real & -M\imag \\ M\imag & M\real
\end{pmatrix} \transpose{Y},
```

where \\(M\real\\) and \\(M\imag\\) are the real and imaginary parts of \\(M\\),
respectively.  If \\(U\\) is an \\(n\times n\\) unitary matrix, then
\\(\UtoSp(U) \transpose{\UtoSp(U)} = \imat[2n]\\) and \\(\UtoSp(U) \sympmat
\transpose{\UtoSp(U)} = \sympmat\\), therefore \\(\UtoSp(U) \in \Sp{2n,\R} \cap
\Og{2n}\\).  Vice versa, if \\(M \in \Sp{2n,\R} \cap \Og{2n}\\) then there exist
\\(A,B \in \Mat{n,\R}\\) such that \\(\transpose{Y} M Y = \begin{psmallmatrix} A
& -B \\\\ B & A \end{psmallmatrix}\\) and \\(A + \iu B \in \Ug{n}\\).  The map
\\(\UtoSp\\) is thus actually a group isomorphism between \\(\Ug{n}\\) and
\\(\Sp{2n,\R} \cap \Og{2n}\\).

Moreover, \\(M \in \Mat{n, \C}\\) is positive-(semi)definite if and only if
\\(\UtoSp(M)\\) is.

#### Williamson decomposition

Let \\(A\\) be a positive-definite \\(2n \times 2n\\) matrix: then there exists
\\(S \in \Sp{2n,\R}\\) and \\(D = \bigoplus\sb{j=1}^n d\sb{j} \imat[2]\\), with
\\(d\sb{j} \geq 1\\), such that \\(A = S D \transpose{S}\\) (cfr. Eq. (3.58) in
[Serafini2023](@cite)).

#### Euler decomposition

Let \\(S \in \Sp{2n,\R}\\): then there exist \\(P,Q \in \Sp{2n,\R} \cap
\Og{2n}\\) and \\(d\sb{1},\dotsc,d\sb{n} > 0\\) such that \\(S = PDQ\\) where
\\(D = \bigoplus\sb{j=1}^{n} \begin{psmallmatrix} d\sb{j} & 0 \\\\ 0 &
d\sb{j}^{-1} \end{psmallmatrix}\\).

We can have the outcome be unitary matrices with the aid of the \\(\UtoSp\\) map
introduced before: given a symplectic \\(2n \times 2n\\) matrix \\(S\\), we can
find \\(U,V \in \Ug{n}\\) such that \\(\UtoSp(U) D \UtoSp(V) = S\\).

### Unitary representation on a symmetric Fock space

The symplectic group of order \\(2n\\) admits a unitary representation
\\(\spunitarysymbol\\) on the symmetric (i.e. bosonic) Fock space with \\(n\\)
modes, \\(\fockb(\C^n)\\).
Given \\(S \in \Sp{2n, \R}\\), its action on the vector \\(\xpvec\\) of position
and momentum operators is given by

```math
\begin{equation}
    \adj{\spunitary{S}} \xpvec_j \spunitary{S} \defeq
    \sum_{k=1}^{2n} S_{jk} \xpvec_k.
\end{equation}
```

such that the canonical commutation relations are preserved, since

```math
\begin{equation*}
    [%
        \adj{\spunitary{S}} R_j \spunitary{S},
        \adj{\spunitary{S}} R_k \spunitary{S}
    ] =
    \sum_{l=1}^{2n} \sum_{m=1}^{2n} S_{jl} S_{km} \sympmat_{lm} =
    \sympmat_{jk}.
\end{equation*}
```

On the annihilation and creation operators of the \\(j\\)-th mode, \\(a\sb{j} =
\frac{1}{\sqrt{2}}(R\sb{2j-1} + \iu R\sb{2j})\\) and \\(\adj{a\sb{j}} =
\frac{1}{\sqrt{2}}(R\sb{2j-1} - \iu R\sb{2j})\\), it behaves as

```math
\begin{multline}
    \adj{\spunitary{S}} a_j \spunitary{S} =
    \frac12 \sum_{k=1}^{n} \bigl[
        (S_{2j-1,2k-1} + S_{2j,2k}) + \iu (S_{2j,2k-1} - S_{2j-1,2k})
    \bigr] a_k +\\+ \frac12 \sum_{k=1}^{n} \bigl[
        (S_{2j-1,2k-1} - S_{2j,2k}) + \iu (S_{2j,2k-1} + S_{2j-1,2k})
    \bigr] \adj{a_k}
\end{multline}
```

and similarly for \\(\adj{\spunitary{S}} \adj{a\sb{j}} \spunitary{S}\\).
If \\(S\\) is also orthogonal, then with \\(M = \UtoSp^{-1}(S)\\) we can also
write

```math
\begin{equation}
  \begin{gathered}
    \adj{\spunitary{\UtoSp(M)}} a_j \spunitary{\UtoSp(M)} =
    \sum_{k=1}^{n} M_{jk} a_k,\\
    \adj{\spunitary{\UtoSp(M)}} \adj{a_j} \spunitary{\UtoSp(M)} =
    \sum_{k=1}^{n} \overline{M_{jk}} \adj{a_k}.
  \end{gathered}
  \label{eq:fock_space_action_symplectic_orthogonal_matrices_shorter}
\end{equation}
```

### Normal-mode decomposition

The covariance matrix can always be diagonalised through a symplectic
transformation: the Williamson decomposition tells us that there always exist
\\(S\in\Sp{2n,\R}\\) and \\(\\{\lambda\sb{j}\\}\sb{j=1}^{n}\\), \\(\lambda\sb{j}
\geq 1\\) such that

```math
\begin{equation}
    \covmat =
    S \biggl( \bigoplus_{j=1}^{n} \lambda_j\imat[2] \biggr) \transpose{S}
\label{eq:covmat-symplectic-diagonalisation}
\end{equation}
```

The _symplectic eigenvalues_ \\(\lambda\sb{j}\\) determine, in turn, the
singular values of the whole state, which can be written as

```math
\begin{equation}
    \gauss{\fmom,\covmat} =
    \displacement{r} \spunitary{S} \Biggl(
        \bigotimes_{j=1}^{n} \biggl(
            \sum_{m=0}^{+\infty} \nu_m(\lambda_j) \outp{\ns{m}}{\ns{m}}
        \biggr)
    \Biggr)
    \adj{\spunitary{S}} \adj{\displacement{r}}
\label{eq:gaussian-state-normal-mode-decomposition}
\end{equation}
```

where \\(\spunitary{S}\\) is the unitary map on \\(\fockb(\C)\tsp{n}\\)
generated by the symplectic matrix \\(S\\) appearing in
\eqref{eq:covmat-symplectic-diagonalisation}, \\(\displacement{r}\\) is the
displacement operator by \\(r\\) and

```math
\nu_m(x) \defeq \frac{2}{x+1}\biggl(\frac{x-1}{x+1}\biggr)^m.
```

We can rewrite this by exchanging tensor product and sum, obtaining

```math
\begin{equation}
    \gauss{\fmom,\covmat} =
    \displacement{r} \spunitary{S} \Biggl(
        \sum_{\multi{m} \in \N^n} \biggl(
            \prod_{j=1}^{n} \nu_{m_j}(\lambda_j)
        \biggr)
        \outp{\ns{\multi{m}}}{\ns{\multi{m}}}
    \Biggr)
    \adj{\spunitary{S}} \adj{\displacement{r}}
\label{eq:normal-mode-decomposition}
\end{equation}
```

or, by defining \\(\hat{\nu}\sb{\multi{m}}(\lambda) \defeq \prod\sb{j=1}^{n}
\nu\sb{m\sb{j}}(\lambda\sb{j})\\),

```math
\begin{equation}
    \gauss{\fmom,\covmat} =
    \sum_{\multi{m} \in \N^n} \hat{\nu}_\multi{m}(\lambda) \,
    \displacement{r} \spunitary{S}
    \outp{\ns{\multi{m}}}{\ns{\multi{m}}}
    \adj{\spunitary{S}} \adj{\displacement{r}}.
\label{eq:gaussian-state-normal-mode-decomposition2}
\end{equation}
```

We can see that the vectors \\(\displacement{r} \spunitary{S} \ns{\multi{m}}\\)
form an orthonormal basis of eigenvectors of \\(\gauss{\fmom,\covmat}\\), with
eigenvalues \\(\hat{\nu}\sb{\multi{m}}(\lambda)\\).  Note that this is _both_ a
diagonalisation and a singular value decomposition (not too obvious in general),
since all the eigenvalues are positive.

!!! info "Ordering the eigenvalues"
    There's no clear way to establish a total order on the eigenvalues: we know
    \\(\nu\sb{m}\\) is decreasing in its argument, and always less than one, for
    each \\(m\\), moreover \\(\nu\sb{m}(x) \leq \nu\sb{m'}(x')\\) for any \\(x,
    x' \geq 1\\) if \\(m' > m\\).
    When we consider \\(\hat{\nu}\\), however, it's more difficult: we can
    only say that \\(\hat{\nu}\sb{\multi{m}}(\lambda) <
    \hat{\nu}\sb{\multi{m}'}(\lambda)\\) if \\(\multi{m} > \multi{m}'\\)
    lexicographically, but still this leaves us with some multiindices we cannot
    compare.
    So, how do we compute the first \\(\chi\\) singular values?

## Construction of the matrix-product state

Suppose we know the covariance matrix \\(\covmat\sb{\psi}\\) and the vector
\\(\fmom\sb{\psi}\\) of first moments of a pure Gaussian state \\(\psi\\) over
\\(M\\) bosonic modes.  We want to find a matrix-product state representation of
\\(\psi\\),

```math
\sum_{m_1=1}^{d} \sum_{m_2=1}^{d} \dotsi \sum_{m_M=1}^{d}
A^{(1,m_1)}
A^{(2,m_2)}
\dotsm
A^{(M,m_M)}
\,
f_{m_1} \otimes f_{m_2} \otimes \dotsb \otimes f_{m_M}
```

in the eigenbasis \\(\ns{\multi{m}}\\) of the number operator.
We will follow the algorithm presented in [Oh2024](@cite).
The full, “true” form of the state

```math
\psi =
\sum_{\multi{m}\in\N^M} c_{m_1,\dotsc,m_M}
\ns{m_1} \otimes \dotsb \otimes \ns{m_M}
```

cannot usually be known, and we will use what we know from
\\(\covmat\sb{\psi}\\) and \\(\fmom\sb{\psi}\\) to reconstruct, to a certain
degree of approximation, the array \\(c\\) of coefficients.
We start by splitting the first mode from the other ones through a Schmidt
decomposition:

```math
\begin{equation}
    \psi =
    \sum_{m\in\N^M} c_{m_1,\dotsc,m_M}
    \ns{m_1} \otimes \dotsb \otimes \ns{m_M} =
    \sum_{\alpha_1=1}^{+\infty} \lambda\modepart{1}_{\alpha_1}
    b\modepart{1}_{\alpha_1} \otimes b\modepart{2\cdots M}_{\alpha_1},
\label{eq:first-schmidt-decomposition}
\end{equation}
```

where \\(\lambda\modepart{1}\sb{\alpha\sb1} \geq 0\\),
\\(\\{b\modepart{1}\sb{\alpha\sb1}\\}\sb{\alpha\sb1}\\) is an orthonormal set in
the Fock space of the first mode, and \\(\\{b\modepart{2\cdots
M}\sb{\alpha\sb1}\\}\sb{\alpha\sb1}\\) is an orthonormal set in the collective
Fock space of modes \\(2\\) to \\(M\\).
We write \\(b\modepart{1}\sb{\alpha\sb1}\\) as a linear combination of the
\\(\ns{m}\\) eigenvectors on the first mode:

```math
b\modepart{1}_{\alpha_1} = \sum_{m_1\in\N}B\modepart{1}_{\alpha_1,m_1} \ns{m_1}
\quad\implies\quad
\psi =
\sum_{\alpha_1=1}^{+\infty} \sum_{m_1\in\N}
\lambda\modepart{1}_{\alpha_1}
B\modepart{1}_{\alpha_1,m_1} \ns{m_1}
\otimes b\modepart{2\cdots M}_{\alpha_1}
```

and identify \\(\lambda\modepart{1}\sb{\alpha\sb1}
B\modepart{1}\sb{\alpha\sb1,m\sb1}\\) with the first tensor of the MPS,
\\(A^{(1,m\sb1)}\sb{\alpha\sb1}\\).  Since both the \\(b\modepart{2\cdots
M}\sb{\alpha\sb1}\\) and the \\(\ns{m\sb1}\\) vectors form eigenbases of their
respective spaces, we have

```math
A^{(1,m_1)}_{\alpha_1} =
\innp{\ns{m_1} \otimes b\modepart{2\cdots M}_{\alpha_1}}{\psi}.
```

We need a way to compute this inner product in a simpler way, since we don't
have access either to the full state \\(\psi\\) or to all eigenvectors
\\(b\modepart{2\cdots M}\sb{\alpha\sb1}\\) from the decomposition
\eqref{eq:first-schmidt-decomposition}.

The first step is using the Williamson decomposition of \\(\psi\\): since it's a
pure state, there exists \\(S\in\Sp{2M,\R}\\) such that
\\(\covmat\sb\psi=S\transpose{S}\\), i.e. \\(\psi = \spunitary{S}\vacuum\\).
We also know that the reduced state \\(\rho\modepart{2\cdots M} = \tr\sb1
\outp{\psi}{\psi}\\) on modes \\(2\\) to \\(M\\) can be written in terms of the
output of the Schmidt decomposition as

```math
\begin{equation}
    \rho\modepart{2\cdots M} =
    \sum_{\alpha_1=1}^{+\infty} (\lambda\modepart{1}_{\alpha_1})^2
    \outp{b\modepart{2\cdots M}_{\alpha_1}}{b\modepart{2\cdots M}_{\alpha_1}}.
\label{eq:svd-reduced-state-2-to-M}
\end{equation}
```

At the same time, we can easily obtain the moments \\(\covmat' \defeq
\covmat\modepart{2\cdots M}\\) and \\(r' \defeq \fmom\modepart{2\cdots M}\\) of
the reduced state by selecting from the original matrix and vector the rows and
columns relative to these modes.  We compute the Williamson decomposition of the
covariance matrix of the reduced state: \\(\covmat' = S' Z \transpose{{S'}}\\),
where \\(Z\\) is a diagonal matrix containing the symplectic eigenvalues
\\(\\{z\sb{j}\\}\sb{j=2}^{M}\\) (which are greater than or equal to \\(1\\)),
and as in Eq. \eqref{eq:normal-mode-decomposition} we write

```math
\begin{equation}
    \rho\modepart{2\cdots M} =
    \sum_{\multi{m} \in \N^{M-1}} \Biggl(\,\prod_{j=2}^{M} \nu_{m_j}(z_j)\Biggr)
    \displacement{r'} \spunitary{S'}
    \outp{%
        \ns{\multi{m}}\modepart{2\cdots M}
    }{%
        \ns{\multi{m}}\modepart{2\cdots M}
    }
    \adj{\spunitary{S'}} \adj{\displacement{r'}}
\label{eq:normal-modes-reduced-state-2-to-M}
\end{equation}
```

Since the state is a positive-semidefinite operator and
\\((\lambda\modepart{1}\sb{\alpha\sb1})^2\\) and \\(\prod\sb{j=2}^{M}
\nu\sb{m\sb{j}}(z\sb{j})\\) are non-negative, both
\eqref{eq:svd-reduced-state-2-to-M} and
\eqref{eq:normal-modes-reduced-state-2-to-M} are the diagonalisation of
\\(\rho\modepart{2\cdots M}\\) on an orthonormal eigenbasis.
The eigenvalue-eigenvector pairs are unique as long as the eigenvalue is not
degenerate, therefore, in that case, we have a one-to-one correspondence

```math
\begin{equation}
    \bigl(
        (\lambda\modepart{1}_{\alpha_1})^2,
        b\modepart{2\cdots M}_{\alpha_1}
    \bigr)
    \Longleftrightarrow
    \Biggl(\,
        \prod_{j=2}^{M} \nu_{m_j}(z_j),
        \displacement{r'} \spunitary{S'} \ns{\multi{m}}\modepart{2\cdots M}
    \Biggr).
\label{eq:svd-reduced-state-correspondence}
\end{equation}
```

Assume we sorted \eqref{eq:svd-reduced-state-2-to-M} so that the terms in the
sum are ordered with decreasing \\((\lambda\sb{\alpha\sb1}\modepart{1})^2\\).
We want to find, from \eqref{eq:svd-reduced-state-correspondence}, the
multi-indices \\(\\{\multi{m}\sb{\alpha\sb1}\\}\sb{\alpha\sb1=1}^{\chi} \subset
\N^{M-1}\\) associated to the \\(\chi\\) largest eigenvalues.
Once we know them, we can proceed to build the first tensor of the MPS as

```math
\begin{equation}
    \begin{split}
    A^{(1,m_1)}_{\alpha_1} &=
    \innp{%
        \ns{m_1} \otimes
        (\displacement{r'} \spunitary{S'} \ns{\multi{m}_{\alpha_1}})
    }{%
        \spunitary{S} \vacuum
    } =\\ &=
    \innp{%
        \ns{m_1} \otimes \ns{\multi{m}_{\alpha_1}}
    }{%
        \adj{\bigl( \id\modepart{1} \otimes
        \displacement{r'}\modepart{2\cdots M}
        \spunitarymp[2\cdots M]{S'} \bigr)}
        \spunitary{S} \vacuum
    }.
    \end{split}
\label{eq:first-mps-tensor}
\end{equation}
```

as in Eq. (3) in [Oh2024](@cite).

!!! todo "How to find the largest eigenvalues"
    We know how to generate all elements on the right-hand side of
    \eqref{eq:svd-reduced-state-correspondence}.
    Assuming there is no degeneracy, the correspondence can be established by
    ordering both sides by decreasing singular value; then, if we want to
    compute the truncated SVD for the MPS compression, we could take the largest
    \\(\chi\\) items.
    But how exactly do we order the singular values?
    Explain how we do this in the package.

From now on, let's ignore displacement operators, since we usually start from
\\(\fmom=0\\).  The symplectic matrix \\(S'\\) on modes \\(2\\) to \\(M\\) can
be extended to all \\(M\\) modes as \\(\imat[2] \oplus S'\\), and in
\eqref{eq:first-mps-tensor} we have

```math
\id\modepart{1} \otimes \spunitarymp[2\cdots M]{S'} =
\spunitary{\imat[2] \oplus S'}
```

The group representation \\(\spunitarysymbol\\) is unitary, so we can
replace \\(\adj{\spunitary{A}}\\) by \\(\spunitary{A^{-1}}\\) and then merge the
two \\(\spunitarysymbol\\)s, obtaining

```math
\adj{ \bigl(\id\modepart{1} \otimes \spunitarymp[2\cdots M]{S'}\bigr) }
\spunitary{S} =
\spunitary{\imat[2] \oplus (S')^{-1}} \spunitary{S} =
\spunitary{(\imat[2] \oplus (S')^{-1}) S};
```

Now, we apply the Euler decompostion to the symplectic matrix \\((\imat[2]
\oplus (S')^{-1}) S\\) to find two new symplectic (and orthogonal) matrices
\\(V\sb1, V\sb2 \in \Sp{2M,\R} \cap \Og{2M}\\) and a diagonal matrix \\(D =
\diag(d\sb1,\dotsc,d\sb{M},d\sb1^{-1},\dotsc,d\sb{M}^{-1})\\) with \\(d\sb{j}
\geq 1\\) such that \\((\imat[2] \oplus (S')^{-1}) S = V\sb1DV\sb2\\).
It follows that

```math
A^{(1,m_1)}_{\alpha_1} =
\innp{%
    \ns{m_1} \otimes \ns{\multi{m}_{\alpha_1}}
}{%
    \spunitary{V_1} \spunitary{D} \spunitary{V_2} \vacuum
}.
```

This inner product can be computed following the algorithm presented in
[Quesada2019](@cite).

!!! todo "To be continued..."
    Explain the next part of the algorithm, in which we compute the remaining
    tensors of the MPS.
