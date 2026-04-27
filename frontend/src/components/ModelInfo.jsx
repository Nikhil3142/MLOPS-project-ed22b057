import React, { useEffect, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

export default function ModelInfo() {
    const [info, setInfo] = useState(null)

    useEffect(() => {
        fetch(`${API_URL}/model-info`)
            .then(r => r.json())
            .then(setInfo)
            .catch(() => { })
    }, [])

    if (!info) return null

    return (
        <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 16px',
            background: 'var(--teal-dim)',
            border: '1px solid rgba(0,194,168,0.2)',
            borderRadius: '999px',
            fontSize: '13px',
        }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: info.model_stage === 'Production' ? 'var(--green)' : 'var(--amber)', display: 'inline-block', flexShrink: 0 }} />
            <span style={{ color: 'var(--gray-400)' }}>
                {info.model_name?.replace(/_/g, ' ')} · v{info.model_version}
                {info.rmse && <span style={{ color: 'var(--teal)', marginLeft: '6px' }}>RMSE: ${Math.round(info.rmse).toLocaleString()}</span>}
            </span>
        </div>
    )
}