name: Task
description: A discrete piece of work
title: "[Task] "
labels: []
assignees: []
body:
  - type: textarea
    id: what
    attributes:
      label: What do you want to do?
      description: "Exact format: I want to [action] to achieve [goal] to benefit the research in [way]."
      placeholder: I want to add a burden penalty to the reward function to achieve more realistic agent behaviour to benefit the research in showing that over-nudging is a real failure mode.
    validations:
      required: true
  - type: textarea
    id: done
    attributes:
      label: Completion criteria
      description: How do we know this is done? What must be true? Use a checklist.
      placeholder: "- [ ] Tests pass\n- [ ] ruff/ty clean\n- [ ] Burden penalty applied when consecutive nudges exceed threshold"
    validations:
      required: true
  - type: textarea
    id: evidence
    attributes:
      label: Evidence
      description: What evidence supports this decision? Link papers, notes, or data.
      placeholder: Trella 2022 Section 3.2; Swapnil/Mengyan 22.06 chat notes
    validations:
      required: false
  - type: textarea
    id: notes
    attributes:
      label: Notes
      description: Dependencies, implementation hints, gotchas
      placeholder: Blocked by #126. See decision-catalogue D10 for burden model options.
    validations:
      required: false
