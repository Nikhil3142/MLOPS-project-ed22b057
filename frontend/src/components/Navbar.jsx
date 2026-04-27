import React from 'react'
import { NavLink } from 'react-router-dom'

const styles = {
    nav: {
        position: 'sticky',
        top: 0,
        zIndex: 100,
        background: 'rgba(10, 22, 40, 0.92)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        padding: '0 40px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '64px',
    },
    logo: {
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
    },
    logoIcon: {
        width: '32px',
        height: '32px',
        background: 'var(--teal)',
        borderRadius: '8px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '16px',
        fontWeight: 700,
        color: 'var(--navy)',
    },
    logoText: {
        fontFamily: 'var(--font-display)',
        fontSize: '20px',
        color: 'var(--white)',
        letterSpacing: '-0.3px',
    },
    logoAccent: {
        color: 'var(--teal)',
    },
    links: {
        display: 'flex',
        gap: '4px',
    },
    badge: {
        fontSize: '11px',
        fontWeight: 600,
        background: 'var(--teal-dim)',
        color: 'var(--teal)',
        padding: '2px 8px',
        borderRadius: '999px',
        border: '1px solid rgba(0,194,168,0.3)',
        letterSpacing: '0.5px',
        textTransform: 'uppercase',
    }
}

export default function Navbar() {
    const linkStyle = ({ isActive }) => ({
        fontFamily: 'var(--font-body)',
        fontWeight: 500,
        fontSize: '14px',
        padding: '8px 18px',
        borderRadius: '8px',
        textDecoration: 'none',
        transition: 'all 0.2s ease',
        color: isActive ? 'var(--teal)' : 'var(--gray-400)',
        background: isActive ? 'var(--teal-dim)' : 'transparent',
        border: isActive ? '1px solid rgba(0,194,168,0.2)' : '1px solid transparent',
    })

    return (
        <nav style={styles.nav}>
            <div style={styles.logo}>
                <div style={styles.logoIcon}>I</div>
                <span style={styles.logoText}>
                    Insure<span style={styles.logoAccent}>AI</span>
                </span>
                <span style={styles.badge}>MLOps</span>
            </div>

            <div style={styles.links}>
                <NavLink to="/predict" style={linkStyle}>
                    🔮 Predict
                </NavLink>
                <NavLink to="/pipeline" style={linkStyle}>
                    ⚙️ Pipeline
                </NavLink>
            </div>
        </nav>
    )
}