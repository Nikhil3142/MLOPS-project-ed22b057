import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Predict from './pages/Predict.jsx'
import Pipeline from './pages/Pipeline.jsx'

export default function App() {
    return (
        <BrowserRouter>
            <Navbar />
            <Routes>
                <Route path="/" element={<Navigate to="/predict" replace />} />
                <Route path="/predict" element={<Predict />} />
                <Route path="/pipeline" element={<Pipeline />} />
            </Routes>
        </BrowserRouter>
    )
}