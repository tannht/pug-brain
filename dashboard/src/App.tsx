import { lazy, Suspense, useEffect } from "react"
import { Routes, Route, Navigate } from "react-router-dom"
import { AppShell } from "@/components/layout/AppShell"
import { PageSkeleton } from "@/components/common/PageSkeleton"
import { useBrainStore } from "@/stores/useBrainStore"
import { useBrains } from "@/api/hooks/useDashboard"

const OverviewPage = lazy(() => import("@/features/overview/OverviewPage"))
const HealthPage = lazy(() => import("@/features/health/HealthPage"))
const GraphPage = lazy(() => import("@/features/graph/GraphPage"))
const TimelinePage = lazy(() => import("@/features/timeline/TimelinePage"))
const EvolutionPage = lazy(() => import("@/features/evolution/EvolutionPage"))
const DiagramsPage = lazy(() => import("@/features/diagrams/DiagramsPage"))
const SettingsPage = lazy(() => import("@/features/settings/SettingsPage"))
const SyncPage = lazy(() => import("@/features/sync/SyncPage"))
const OraclePage = lazy(() => import("@/features/oracle/OraclePage"))

/**
 * PugBrain Dashboard — Auto-detects the 'default' brain on first load.
 * Gâu gâu! 🐶
 */
export default function App() {
  const { setActiveBrain, initialized, setInitialized } = useBrainStore()
  const { data: brains } = useBrains()

  // Auto-detect active brain on first load
  useEffect(() => {
    if (initialized || !brains || brains.length === 0) return

    // Find the currently active brain from server
    const active = brains.find((b) => b.is_active)
    if (active) {
      setActiveBrain(active.name)
    } else {
      // Fallback: try "default" brain, or just pick the first one
      const defaultBrain = brains.find((b) => b.name === "default")
      setActiveBrain(defaultBrain?.name ?? brains[0].name)
    }
    setInitialized(true)
  }, [brains, initialized, setActiveBrain, setInitialized])

  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route
          index
          element={
            <Suspense fallback={<PageSkeleton />}>
              <OverviewPage />
            </Suspense>
          }
        />
        <Route
          path="health"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <HealthPage />
            </Suspense>
          }
        />
        <Route
          path="graph"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <GraphPage />
            </Suspense>
          }
        />
        <Route
          path="timeline"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <TimelinePage />
            </Suspense>
          }
        />
        <Route
          path="evolution"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <EvolutionPage />
            </Suspense>
          }
        />
        <Route
          path="diagrams"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <DiagramsPage />
            </Suspense>
          }
        />
        <Route
          path="settings"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <SettingsPage />
            </Suspense>
          }
        />
        <Route
          path="sync"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <SyncPage />
            </Suspense>
          }
        />
        <Route
          path="oracle"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <OraclePage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
