import React from 'react'

export default function CostCard({ result }) {
    const { predicted_charge, model_name, model_version, latency_ms } = result

    const formatCurrency = (val) =>
        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val)

    const getRiskLevel = (charge) => {
        if (charge < 8000) return { label: 'Low Risk', color: '#10b981', bg: 'rgba(16,185,129,0.12)' }
        if (charge < 20000) return { label: 'Medium Risk', color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' }
        return { label: 'High Risk', color: '#ef4444', bg: 'rgba(239,68,68,0.12)' }
    }

    const risk = getRiskLevel(predicted_charge)

    return (
        <div className="card fade-up" style={{ textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
            {/* Background glow */}
            <div style={{
                position: 'absolute', top: '-60px', left: '50%', transform: 'translateX(-50%)',
                width: '200px', height: '200px',
                background: 'radial-gradient(circle, rgba(0,194,168,0.12) 0%, transparent 70%)',
                pointerEvents: 'none',
            }} />

            <div style={{ position: 'relative' }}>
                <p style={{ fontSize: '13px', fontWeight: 500, color: 'var(--gray-400)', letterSpacing: '1px', textTransform: 'uppercase', marginBottom: '12px' }}>
                    Estimated Annual Premium
                </p>

                <div style={{ fontSize: '56px', fontFamily: 'var(--font-display)', color: 'var(--teal)', lineHeight: 1, marginBottom: '16px', letterSpacing: '-2px' }}>
                    {formatCurrency(predicted_charge)}
                </div>

                <div style={{
                    display: 'inline-block',
                    padding: '6px 20px',
                    borderRadius: '999px',
                    background: risk.bg,
                    color: risk.color,
                    fontWeight: 600,
                    fontSize: '13px',
                    border: `1px solid ${risk.color}40`,
                    marginBottom: '24px',
                }}>
                    {risk.label}
                </div>

                <div style={{ display: 'flex', justifyContent: 'center', gap: '32px', borderTop: '1px solid rgba(255,255,255,0.07)', paddingTop: '20px' }}>
                    <div>
                        <p style={{ fontSize: '11px', color: 'var(--gray-400)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Model</p>
                        <p style={{ fontSize: '14px', fontWeight: 600, color: 'var(--white)', textTransform: 'capitalize' }}>{model_name?.replace(/_/g, ' ')}</p>
                    </div>
                    <div>
                        <p style={{ fontSize: '11px', color: 'var(--gray-400)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Version</p>
                        <p style={{ fontSize: '14px', fontWeight: 600, color: 'var(--white)' }}>v{model_version}</p>
                    </div>
                    <div>
                        <p style={{ fontSize: '11px', color: 'var(--gray-400)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Latency</p>
                        <p style={{ fontSize: '14px', fontWeight: 600, color: latency_ms < 200 ? 'var(--green)' : 'var(--red)' }}>{latency_ms?.toFixed(1)}ms</p>
                    </div>
                </div>
            </div>
        </div>
    )
}