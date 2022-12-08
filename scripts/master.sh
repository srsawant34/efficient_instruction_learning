# !/bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--taskfile)
      TASKFILE="$2"
      shift 
      shift 
      ;;
    -s|--sampletype)
      SAMPLETYPE="$2"
      shift 
      shift 
      ;;
    -e|--experiement)
      EXPERIMENT="$2"
      shift
      shift 
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") 
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"
if [[ "${TASKFILE: -4}" == ".txt" ]]; then
    echo "Taskfile: $TASKFILE"
else
    echo "Provide correct task file."
    exit 1
fi
if [[ "$SAMPLETYPE" =~ ^(ten|onepercent|hundred|twohundred|thousand)$ ]]; then
    echo "Sample type: $SAMPLETYPE"
else
    echo "Incorrect sample type- $SAMPLETYPE"
    exit 2
fi
if [[ "$EXPERIMENT" =~ ^(crosstask|)$ ]]; then
    if [[ "$EXPERIMENT" =~ ^()$ ]]; then
        echo ""
    else
        echo "Experiment: $EXPERIMENT"
    fi
else
    echo "Incorrect experiment name $EXPERIMENT"
    exit 3
fi

sh scripts/train.sh $SAMPLETYPE $TASKFILE $EXPERIMENT && sh scripts/eval.sh $SAMPLETYPE $TASKFILE $EXPERIMENT;
exit 0