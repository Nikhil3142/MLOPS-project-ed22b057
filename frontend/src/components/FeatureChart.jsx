import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer, ReferenceLine } from 'recharts'

const FEATURE_LABELS = {
    age: 'Age',
    sex: 'Sex',
    bmi: 'BMI',
    children: 'Children',
    smoker: 'Smoker',
    region_northwest: 'Region NW',
    region_southeast: 'Region SE',
    region_southwest: 'Region SW',
}

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const val = payload[0].value
        return (
            <div style={{
                background: 'var(--navy)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                padding: '10px 14px',
                fontSize: '13px',
            }}>
                <p style={{ color: 'var(--gray-400)', marginBottom: '4px' }}>{payload[0].payload.feature}</p>
                <p style={{ color: val >= 0 ? 'var(--teal)' : 'var(--red)', fontWeight: 600 }}>
                    {val >= 0 ? '+' : ''}{new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val)}
                </p>
            </div>
        )
    }
    return null
}

export default function FeatureChart({ importance }) {
    const data = Object.entries(importance)
        .map(([key, value]) => ({
            feature: FEATURE_LABELS[key] || key,
            value: Math.round(value),
            abs: Math.abs(value),
        }))
        .sort((a, b) => b.abs - a.abs)

    return (
        <div className="card fade-up-delay-1">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                <div>
                    <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '20px', marginBottom: '4px' }}>Feature Impact</h3>
                    <p style={{ fontSize: '13px', color: 'var(--gray-400)' }}>SHAP values — how each factor affects your premium</p>
                </div>
                <span className="tag tag-teal">SHAP</span>
            </div>

            <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, left: 60, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis
                        type="number"
                        tick={{ fill: 'var(--gray-400)', fontSize: 12 }}
                        tickFormatter={(v) => `$${Math.round(v / 1000)}k`}
                        axisLine={false}
                        tickLine={false}
                    />
                    <YAxis
                        type="category"
                        dataKey="feature"
                        tick={{ fill: 'var(--gray-400)', fontSize: 12 }}
                        axisLine={false}
                        tickLine={false}
                        width={60}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                    <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
                        {data.map((entry, index) => (
                            <Cell
                                key={index}
                                fill={entry.value >= 0 ? 'var(--teal)' : '#ef4444'}
                                fillOpacity={0.85}
                            />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>

            <p style={{ fontSize: '12px', color: 'var(--gray-600)', marginTop: '12px', textAlign: 'center' }}>
                Positive values increase premium · Negative values decrease premium
            </p>
        </div>
    )
}