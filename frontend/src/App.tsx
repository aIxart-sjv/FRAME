import { TopBar } from './components/layout/TopBar'
import { LandingSection } from './components/landing/LandingSection'
import { ResultsView } from './components/results/ResultsView'
import { useBackendHealth } from './hooks/useBackendHealth'
import { useFrameSession } from './state/useFrameSession'
import './App.css'

function App() {
  const session = useFrameSession()
  const { status, health } = useBackendHealth() // polled once here; shared by the top bar and the model selector

  return (
    <div className="app-shell">
      <TopBar status={status} health={health} />
      <main className="app-shell__main">
        {session.hasResult ? (
          <>
            <div className="app-shell__toolbar">
              <p className="app-shell__toolbar-label mono">
                Analyzing <strong>{session.file?.name}</strong>
              </p>
              <button type="button" className="app-shell__reset-button" onClick={session.reset}>
                New analysis
              </button>
            </div>
            <ResultsView session={session} />
          </>
        ) : (
          <LandingSection session={session} health={health} />
        )}
      </main>
    </div>
  )
}

export default App
