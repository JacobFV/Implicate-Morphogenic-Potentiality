if [[ "$1" == "-w" ]] || [[ "$1" == "--watch" ]]; then
  while true; do
    latexmk -pdf -pvc main.tex
    inotifywait -e modify *.tex
  done
else
  latexmk -pdf -pvc main.tex
fi
