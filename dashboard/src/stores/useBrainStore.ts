import { create } from "zustand"

interface BrainState {
  activeBrain: string | null
  setActiveBrain: (name: string) => void
  initialized: boolean
  setInitialized: (v: boolean) => void
}

export const useBrainStore = create<BrainState>((set) => ({
  activeBrain: null,
  initialized: false,
  setActiveBrain: (name) => set({ activeBrain: name }),
  setInitialized: (v) => set({ initialized: v }),
}))
