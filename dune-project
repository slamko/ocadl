(lang dune 3.6)

(name ocadl)

(generate_opam_files true)

(source
 (github username/reponame))

(authors "Viacheslav Chepelyk-Kozhin")

(maintainers "Viacheslav Chepelyk-Kozhin")

(license GPL-v3.0)

(documentation https://url/to/documentation)

(package
 (name ocadl)
 (synopsis "A short synopsis")
 (description "A longer description")
 (depends ocaml dune)
 (tags
  (topics "to describe" your project)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
