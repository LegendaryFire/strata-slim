# patches/

Your own changes to upstream code. `extract.sh` applies every `*.patch` here, in name order, on top of the
extracted tag. A patch that no longer applies stops the extract and names itself, so a change is never silently lost.

Make one from a change in `vendor/`:

    git diff --relative=vendor -- vendor/ > patches/010-what-it-does.patch

(`--relative=vendor` writes the paths as upstream has them, which is what `extract.sh` expects.) Then run
`./extract.sh <same tag>` once: vendor/ comes out identical, which proves the patch applies.
