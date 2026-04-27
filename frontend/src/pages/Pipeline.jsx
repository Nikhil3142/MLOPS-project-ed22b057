import React, { useState, useEffect } from 'react'
import ModelInfo from '../components/ModelInfo.jsx'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'
const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3001'
const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL || 'http://localhost:5000'
const AIRFLOW_URL = import.meta.env.VITE_AIRFLOW_URL || 'http://localhost:8081'

const tabStyle = (active) => ({
    padding: '10px 20px',
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    fontFamily: 'var(--font-body)',
    fontWeight: 500,
    fontSize: '14px',
    transition: 'all 0.2s ease',
    background: active ? 'var(--teal-dim)' : 'transparent',
    color: active ? 'var(--teal)' : 'var(--gray-400)',
    borderBottom: active ? '2px solid var(--teal)' : '2px solid transparent',
})

const TABS = [
    { id: 'grafana', label: '📊 Monitoring', url: `${GRAFANA_URL}/d/insurance-mlops-v1` },
    { id: 'mlflow', label: '🧪 Experiments', url: `${MLFLOW_URL}/#/experiments/1` },
    { id: 'airflow', label: '🔄 Orchestration', url: `${AIRFLOW_URL}/dags/ingestion_dag/grid` },
]

function StatusDot({ ok }) {
    return (
        <span style={{
            display: 'inline-block',
            width: '8px', height: '8px',
            borderRadius: '50%',
            background: ok ? 'var(--green)' : 'var(--red)',
            boxShadow: ok ? '0 0 6px var(--green)' : '0 0 6px var(--red)',
            flexShrink: 0,
        }} />
    )
}

export default function Pipeline() {
    const [activeTab, setActiveTab] = useState('grafana')
    const [metrics, setMetrics] = useState(null)
    const [modelInfo, setModelInfo] = useState(null)
    const [reloading, setReloading] = useState(false)
    const [reloadMsg, setReloadMsg] = useState(null)

    useEffect(() => {
        // Fetch model info
        fetch(`${API_URL}/model-info`)
            .then(r => r.json())
            .then(setModelInfo)
            .catch(() => { })

        // Fetch health
        fetch(`${API_URL}/health`)
            .then(r => r.json())
            .then(d => setMetrics(d))
            .catch(() => { })
    }, [])

    const handleReload = async () => {
        setReloading(true)
        setReloadMsg(null)
        try {
            const res = await fetch(`${API_URL}/reload`, { method: 'POST' })
            const data = await res.json()
            if (res.ok) {
                setReloadMsg({ ok: true, text: `Model reloaded — v${data.model_version}` })
                const info = await fetch(`${API_URL}/model-info`).then(r => r.json())
                setModelInfo(info)
            } else {
                setReloadMsg({ ok: false, text: data.detail || 'Reload failed' })
            }
        } catch {
            setReloadMsg({ ok: false, text: 'Could not reach backend' })
        } finally {
            setReloading(false)
        }
    }

    const activeUrl = TABS.find(t => t.id === activeTab)?.url

    return (
        <div style={{ minHeight: 'calc(100vh - 64px)', padding: '40px', maxWidth: '1400px', margin: '0 auto' }}>

            {/* Header */}
            <div className="fade-up" style={{ marginBottom: '32px' }}>
                <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '42px', lineHeight: 1.1, marginBottom: '8px', letterSpacing: '-1px' }}>
                    ML Pipeline<br />
                    <span style={{ color: 'var(--teal)' }}>Control Centre</span>
                </h1>
                <p style={{ color: 'var(--gray-400)', fontSize: '15px' }}>
                    Monitor your pipeline, experiments, and orchestration in real time.
                </p>
            </div>

            {/* Status Bar */}
            <div className="fade-up-delay-1" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '32px' }}>
                {[
                    { label: 'Backend API', ok: !!metrics, value: metrics ? 'Online' : 'Offline' },
                    { label: 'Model Loaded', ok: modelInfo?.model_stage === 'Production', value: modelInfo?.model_stage === 'Production' ? 'Ready' : 'Not Loaded' },
                    { label: 'Active Model', ok: true, value: modelInfo?.model_name?.replace(/_/g, ' ') || '—' },
                    { label: 'Model Version', ok: true, value: modelInfo?.model_version ? `v${modelInfo.model_version}` : '—' },
                ].map((s, i) => (
                    <div key={i} className="card" style={{ padding: '18px 20px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                            <StatusDot ok={s.ok} />
                            <span style={{ fontSize: '12px', color: 'var(--gray-400)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>{s.label}</span>
                        </div>
                        <p style={{ fontSize: '18px', fontWeight: 600, textTransform: 'capitalize' }}>{s.value}</p>
                    </div>
                ))}
            </div>

            {/* Reload button */}
            <div className="fade-up-delay-2" style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
                <button
                    className="btn btn-primary"
                    onClick={handleReload}
                    disabled={reloading}
                    style={{ padding: '10px 24px', fontSize: '14px' }}
                >
                    {reloading ? (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className="spinner" style={{ borderTopColor: 'var(--navy)' }} /> Reloading...
                        </span>
                    ) : '↻ Reload Production Model'}
                </button>

                {reloadMsg && (
                    <span style={{
                        fontSize: '13px',
                        color: reloadMsg.ok ? 'var(--green)' : 'var(--red)',
                        padding: '8px 14px',
                        background: reloadMsg.ok ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
                        borderRadius: '8px',
                        border: `1px solid ${reloadMsg.ok ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
                    }}>
                        {reloadMsg.ok ? '✓' : '✗'} {reloadMsg.text}
                    </span>
                )}
            </div>

            {/* Tabs */}
            <div className="fade-up-delay-2" style={{ display: 'flex', gap: '4px', marginBottom: '0', borderBottom: '1px solid rgba(255,255,255,0.07)', paddingBottom: '0' }}>
                {TABS.map(tab => (
                    <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={tabStyle(activeTab === tab.id)}>
                        {tab.label}
                    </button>
                ))}
                <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center' }}>
                    <a
                        href={activeUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ fontSize: '13px', color: 'var(--teal)', textDecoration: 'none', padding: '8px 12px' }}
                    >
                        Open in new tab ↗
                    </a>
                </div>
            </div>

            {/* iframe */}
            <div className="fade-up-delay-3" style={{ borderRadius: '0 0 var(--radius-lg) var(--radius-lg)', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.07)', borderTop: 'none' }}>
                <iframe
                    key={activeTab}
                    src={activeUrl}
                    style={{ width: '100%', height: '680px', border: 'none', background: '#fff' }}
                    title={activeTab}
                />
            </div>

            {/* Pipeline stages */}
            <div className="fade-up-delay-3" style={{ marginTop: '32px' }}>
                <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '22px', marginBottom: '20px' }}>DVC Pipeline Stages</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                    {[
                        { stage: '01', name: 'Preprocess', desc: 'Train/test split + StandardScaler', icon: '⚙️' },
                        { stage: '02', name: 'Train', desc: 'Dynamic model loading via params.yaml', icon: '🧠' },
                        { stage: '03', name: 'Evaluate', desc: 'RMSE, MAE, R², latency, SHAP values', icon: '📊' },
                        { stage: '04', name: 'Register', desc: 'Best run → MLflow Model Registry', icon: '📦' },
                    ].map((s, i) => (
                        <div key={i} className="card" style={{ padding: '20px', position: 'relative', overflow: 'hidden' }}>
                            <div style={{ position: 'absolute', top: '12px', right: '16px', fontSize: '11px', fontWeight: 700, color: 'var(--teal)', opacity: 0.4, fontFamily: 'monospace' }}>
                                {s.stage}
                            </div>
                            <div style={{ fontSize: '24px', marginBottom: '10px' }}>{s.icon}</div>
                            <h4 style={{ fontWeight: 600, marginBottom: '6px' }}>{s.name}</h4>
                            <p style={{ fontSize: '12px', color: 'var(--gray-400)', lineHeight: 1.5 }}>{s.desc}</p>
                            {i < 3 && (
                                <div style={{ position: 'absolute', right: '-8px', top: '50%', transform: 'translateY(-50%)', zIndex: 1, color: 'var(--teal)', fontSize: '16px' }}>→</div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}