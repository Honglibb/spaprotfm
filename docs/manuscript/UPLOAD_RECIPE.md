# Release recipe: GitHub repo + Zenodo archive

Step-by-step commands for publishing SpaProtFM code and pretrained
checkpoints before submission. Run from your **local laptop** unless
otherwise noted.

---

## 1. GitHub: create the public code repo

**A. On github.com**, sign in and create a new empty repository named
`spaprotfm` (or any name you prefer — keep a note of it, used below
as `<REPO>`). Do **not** initialise it with README / license / gitignore
(we already have those locally).

**B. In your ssh session to zkgy**, from the project root
`/home/zkgy/hongliyin_computer`:

```bash
# Add the GitHub remote and push main
git remote add origin https://github.com/Honglibb/spaprotfm.git
git branch -M main
git push -u origin main
```

GitHub will ask for a **personal access token** (classic) the first
time. If you need one, generate at
`https://github.com/settings/tokens` → scope = `repo`.

After push, your repo URL is:
```
https://github.com/Honglibb/spaprotfm
```

Tell me this URL and I'll substitute it into the manuscript's
Data Availability section.

---

## 2. Zenodo: upload the checkpoints

Zenodo is CERN-backed, free, and issues a **permanent DOI** that can
be cited in the paper. Bioinformatics journals accept Zenodo DOIs for
model / data release.

**A.** Go to `https://zenodo.org/` and sign in with GitHub or ORCID
(use the same ORCID you'll list on the manuscript).

**B.** Download the bundle from the server to your laptop:

```bash
# on your local laptop
scp zkgy:~/Desktop/spaprotfm_v2_checkpoints.zip ~/Desktop/
```

The zip is 70 MB and contains four checkpoint directories plus
`DATA_README.md` (readable during review before download).

**C.** On Zenodo, click **New Upload** and fill in:

| Field | Value |
|---|---|
| Title | SpaProtFM v2: pretrained checkpoints for cross-cohort IMC panel extension |
| Upload type | Dataset (or Software — both work; Dataset is cleaner for weights) |
| Authors | Hongli Yin — ORCID iD (optional) — affiliation: Institute of Pediatric Research, Children's Hospital of Soochow University |
| Description | (copy in the first paragraph of `DATA_README.md` from inside the zip) |
| License | CC-BY-4.0 |
| Keywords | imaging mass cytometry, spatial proteomics, panel extension, foundation model, Phikon-v2, pathology |
| Related identifiers | `https://github.com/Honglibb/spaprotfm` → relation: "is supplemented by" |
| Version | v1.0 |

Drag and drop the `spaprotfm_v2_checkpoints.zip` → click **Publish**.

Zenodo will give you a DOI like `10.5281/zenodo.123456789`. Tell me
this DOI and I'll substitute it into the manuscript.

---

## 3. (Optional) Hugging Face for model-card discoverability

If you want Phikon-style "model card" discoverability — so other
researchers searching Hugging Face for "IMC panel extension" can find
your checkpoints — repeat the upload at `https://huggingface.co/new`
under a repo like `Honglibb/spaprotfm-v2`. Not required for BIB
submission; the Zenodo DOI is sufficient and more formal.

---

## 4. What goes in the manuscript after you finish the uploads

Once you give me:

1. GitHub URL (from §1)
2. Zenodo DOI (from §2)

I will:

- replace `[github url TBD]` in the README with the real URL;
- replace `[repository TBD]` and `[DOI TBD]` in the manuscript's
  Data Availability section with the Zenodo DOI;
- update `@article{yin2026spaprotfm, ...}` BibTeX stub in the README
  with the final `howpublished` / `url` fields;
- re-commit and push so the release is reproducible from tag.

---

## 5. Suggested release tag

Once the Zenodo DOI and GitHub URL are in place, tag the code:

```bash
git tag -a v1.0 -m "Release accompanying SpaProtFM v2 manuscript (BIB submission)"
git push origin v1.0
```

Zenodo can be configured to **auto-archive future tagged releases**
of the GitHub repo — enable under the Zenodo ↔ GitHub integration
once you've published v1.0.
