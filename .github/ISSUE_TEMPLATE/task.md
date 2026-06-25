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
      description: One sentence: I want to X
      placeholder: I want to add a burden penalty to the reward function
    validations:
      required: true
  - type: textarea
    id: why
    attributes:
      label: Why?
      description: How does this benefit the research?
      placeholder: So that the agent learns to avoid over-nudging users, which is a known failure mode in JITAIs
    validations:
      required: true
  - type: textarea
    id: done
    attributes:
      label: Completion criteria
      description: How do we know this is done? What must be true?
      placeholder: Tests pass, ruff/ty clean, burden penalty is applied when consecutive nudges exceed threshold
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
