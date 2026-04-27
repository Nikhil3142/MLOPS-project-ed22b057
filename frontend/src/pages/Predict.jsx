import React, { useState } from 'react'
import CostCard from '../components/CostCard.jsx'
import FeatureChart from '../components/FeatureChart.jsx'
import ModelInfo from '../components/ModelInfo.jsx'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

const INITIAL_FORM = {
    age: '',
    sex: 'male',
    bmi: '',
    children: '0',
    smoker: 'no',
    region: 'northeast',
}

const inputStyle = {
    width: '100%',
    padding: '12px 16px',
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '10px',
    color: 'var(--white)',
    fontSize: '15px',
    fontFamily: 'var(--font-body)',
    outline: 'none',
    transition: 'all 0.2s ease',
}

const labelStyle = {
    display: 'block',
    fontSize: '12px',
    fontWeight: 600,
    color: 'var(--gray-400)',
    textTransform: 'uppercase',
    letterSpacing: '0.8px',
    marginBottom: '8px',
}

function Field({ label, children }) {
    return (
        <div>
            <label style={labelStyle}>{label}</label>
            {children}
        </div>
    )
}

function SelectField({ label, value, onChange, options }) {
    return (
        <Field label={label}>
            <select value={value} onChange={onChange} style={{ ...inputStyle, cursor: 'pointer' }}>
                {options.map(o => (
                    <option key={o.value} value={o.value} style={{ background: 'var(--navy-light)' }}>
                        {o.label}
                    </option>
                ))}
            </select>
        </Field>
    )
}

export default function Predict() {
    const [form, setForm] = useState(INITIAL_FORM)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const set = (key) => (e) => setForm(f => ({ ...f, [key]: e.target.value }))

    const handleSubmit = async () => {
        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const payload = {
                age: parseInt(form.age),
                sex: form.sex,
                bmi: parseFloat(form.bmi),
                children: parseInt(form.children),
                smoker: form.smoker,
                region: form.region,
            }

            const res = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Prediction failed')
            }

            const data = await res.json()
            setResult(data)
        } catch (e) {
            setError(e.message)
        } finally {
            setLoading(false)
        }
    }

    const isValid = form.age && form.bmi && parseInt(form.age) >= 18 && parseFloat(form.bmi) > 0

    return (
        <div style={{ minHeight: 'calc(100vh - 64px)', padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>

            {/* Header */}
            <div className="fade-up" style={{ marginBottom: '40px' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
                    <div>
                        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '42px', lineHeight: 1.1, marginBottom: '8px', letterSpacing: '-1px' }}>
                            Insurance Cost<br />
                            <span style={{ color: 'var(--teal)' }}>Predictor</span>
                        </h1>
                        <p style={{ color: 'var(--gray-400)', fontSize: '15px', maxWidth: '440px' }}>
                            Enter your details to get an AI-powered estimate of your annual medical insurance premium.
                        </p>
                    </div>
                    <ModelInfo />
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', alignItems: 'start' }}>

                {/* Form */}
                <div className="card fade-up-delay-1">
                    <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '22px', marginBottom: '24px' }}>
                        Your Information
                    </h2>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                        <Field label="Age (years)">
                            <input
                                type="number"
                                min="18" max="100"
                                placeholder="e.g. 35"
                                value={form.age}
                                onChange={set('age')}
                                style={inputStyle}
                                onFocus={e => e.target.style.borderColor = 'var(--teal)'}
                                onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
                            />
                        </Field>

                        <Field label="BMI">
                            <input
                                type="number"
                                min="10" max="60"
                                step="0.1"
                                placeholder="e.g. 28.5"
                                value={form.bmi}
                                onChange={set('bmi')}
                                style={inputStyle}
                                onFocus={e => e.target.style.borderColor = 'var(--teal)'}
                                onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
                            />
                        </Field>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                        <SelectField
                            label="Sex"
                            value={form.sex}
                            onChange={set('sex')}
                            options={[
                                { value: 'male', label: '♂ Male' },
                                { value: 'female', label: '♀ Female' },
                            ]}
                        />
                        <SelectField
                            label="Smoker"
                            value={form.smoker}
                            onChange={set('smoker')}
                            options={[
                                { value: 'no', label: '🚭 Non-smoker' },
                                { value: 'yes', label: '🚬 Smoker' },
                            ]}
                        />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
                        <SelectField
                            label="Dependents"
                            value={form.children}
                            onChange={set('children')}
                            options={[0, 1, 2, 3, 4, 5].map(n => ({ value: String(n), label: `${n} ${n === 1 ? 'child' : 'children'}` }))}
                        />
                        <SelectField
                            label="Region"
                            value={form.region}
                            onChange={set('region')}
                            options={[
                                { value: 'northeast', label: '🗺 Northeast' },
                                { value: 'northwest', label: '🗺 Northwest' },
                                { value: 'southeast', label: '🗺 Southeast' },
                                { value: 'southwest', label: '🗺 Southwest' },
                            ]}
                        />
                    </div>

                    {/* BMI Guide */}
                    <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '8px', padding: '12px 16px', marginBottom: '24px', fontSize: '12px', color: 'var(--gray-400)' }}>
                        <strong style={{ color: 'var(--gray-400)' }}>BMI Guide:</strong>{' '}
                        Underweight &lt;18.5 · Normal 18.5–24.9 · Overweight 25–29.9 · Obese ≥30
                    </div>

                    <button
                        className="btn btn-primary"
                        onClick={handleSubmit}
                        disabled={loading || !isValid}
                        style={{ width: '100%', fontSize: '16px', padding: '16px' }}
                    >
                        {loading ? (
                            <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                                <span className="spinner" /> Calculating...
                            </span>
                        ) : '✦ Predict My Premium'}
                    </button>

                    {error && (
                        <div style={{ marginTop: '16px', padding: '12px 16px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', color: '#ef4444', fontSize: '14px' }}>
                            ⚠️ {error}
                        </div>
                    )}
                </div>

                {/* Results */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    {result ? (
                        <>
                            <CostCard result={result} />
                            <FeatureChart importance={result.feature_importance} />
                        </>
                    ) : (
                        <div className="card fade-up-delay-2" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '320px', textAlign: 'center', border: '1px dashed rgba(255,255,255,0.1)' }}>
                            <div style={{ fontSize: '48px', marginBottom: '16px' }}>🏥</div>
                            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '20px', marginBottom: '8px', color: 'var(--gray-400)' }}>
                                Your prediction will appear here
                            </h3>
                            <p style={{ fontSize: '14px', color: 'var(--gray-600)', maxWidth: '260px' }}>
                                Fill in your details and click predict to get your estimated annual premium.
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Info cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginTop: '40px' }}>
                {[
                    { icon: '🤖', title: 'AI-Powered', desc: 'XGBoost model trained on 1,338 insurance records with 5-fold validation' },
                    { icon: '⚡', title: 'Real-time', desc: 'Sub-200ms inference powered by FastAPI and MLflow model registry' },
                    { icon: '🔍', title: 'Explainable', desc: 'SHAP values show exactly which factors drive your premium estimate' },
                ].map((c, i) => (
                    <div key={i} className={`card fade-up-delay-${i + 1}`} style={{ padding: '20px' }}>
                        <div style={{ fontSize: '28px', marginBottom: '10px' }}>{c.icon}</div>
                        <h4 style={{ fontWeight: 600, marginBottom: '6px' }}>{c.title}</h4>
                        <p style={{ fontSize: '13px', color: 'var(--gray-400)', lineHeight: 1.5 }}>{c.desc}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}