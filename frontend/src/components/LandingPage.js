import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import LocomotiveScroll from 'locomotive-scroll';
import './LandingPage.css';

gsap.registerPlugin(ScrollTrigger);

function LandingPage() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const loaderRef = useRef(null);
  const contentRef = useRef(null);

  useEffect(() => {
    // Initialize Locomotive Scroll for the Landing Page
    const locoScroll = new LocomotiveScroll({
      lenisOptions: {
        lerp: 0.1,
        smoothWheel: true,
      }
    });

    // Loading Screen Animation
    const tl = gsap.timeline({
      onComplete: () => {
        setIsLoading(false);
      }
    });

    if (loaderRef.current) {
      tl.to(loaderRef.current.querySelector('.loader-progress'), {
        width: '100%',
        duration: 1.5,
        ease: 'power3.inOut'
      })
      .to(loaderRef.current, {
        opacity: 0,
        duration: 0.6,
        ease: 'power2.out'
      }, "+=0.2");
    }

    return () => {
      locoScroll.destroy();
    };
  }, []);

  useEffect(() => {
    if (!isLoading && contentRef.current) {
      // Entrance Animation for Landing Content
      gsap.fromTo('.hero-anim', 
        { y: 50, opacity: 0 }, 
        { y: 0, opacity: 1, duration: 1, stagger: 0.15, ease: 'power3.out', delay: 0.2 }
      );

      // Scroll animations
      gsap.utils.toArray('.scroll-fade-up').forEach(element => {
        gsap.fromTo(element,
          { y: 50, opacity: 0 },
          {
            y: 0,
            opacity: 1,
            duration: 0.8,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: element,
              start: "top 85%",
            }
          }
        );
      });
    }
  }, [isLoading]);

  const handleEnter = () => {
    gsap.to(contentRef.current, {
      opacity: 0,
      y: -50,
      duration: 0.6,
      ease: 'power2.in',
      onComplete: () => navigate('/app')
    });
  };

  return (
    <>
      {isLoading && (
        <div className="landing-loader" ref={loaderRef}>
          <div className="loader-brand">ACRUX_CORE_INIT</div>
          <div className="loader-bar-container">
            <div className="loader-progress"></div>
          </div>
        </div>
      )}
      
      <div className="acrux-landing" ref={contentRef} style={{ opacity: isLoading ? 0 : 1 }}>
        <div className="landing-nav">
          <div className="nav-logo">ACRUX</div>
          <div className="nav-status"><span className="orange-dot">●</span> UPLINK STABLE</div>
        </div>

        {/* HERO SECTION */}
        <section className="hero-section">
          <div className="hero-text" data-scroll data-scroll-speed="2">
            <div className="system-online hero-anim"><span className="orange-dot">●</span> SYSTEM ONLINE</div>
            <h1 className="hero-anim">
              ACRUX at the<br/>
              <span className="text-orange">Speed of Sight.</span>
            </h1>
            <p className="hero-anim">
              Standard NLP is blind. ACRUX fuses linguistic, visual, and kinetic data streams into a unified truth matrix, delivering explainable insight in milliseconds.
            </p>
            <button className="cyber-btn primary hero-anim mt-4" onClick={handleEnter}>
              INITIALIZE ACRUX <span className="arrow">→</span>
            </button>
          </div>
          <div className="hero-visual hero-anim" data-scroll data-scroll-speed="1">
            <div className="orb-container">
              <div className="core-orb"></div>
              <div className="ring-1"></div>
              <div className="ring-2"></div>
            </div>
          </div>
        </section>

        {/* CONTEXT SECTION */}
        <section className="context-section scroll-fade-up">
          <h2 className="section-title" data-scroll data-scroll-speed="1.5">Context is the <span className="text-orange">Ghost in the Machine.</span></h2>
          <p className="section-subtitle" data-scroll data-scroll-speed="1">
            Words tell half the story. Tone, expression, and environment tell the rest. Standard models miss the nuance. ACRUX captures it all.
          </p>
          
          <div className="comparison-grid">
            <div className="comp-card dark" data-scroll data-scroll-speed="2">
              <div className="comp-header">Standard NLP</div>
              <div className="comp-row"><span className="label">Text Parsing</span><span className="val">20%</span></div>
              <div className="comp-row"><span className="label">Sarcasm Detection</span><span className="val">LOW</span></div>
              <div className="comp-row"><span className="label">Visual Context</span><span className="val">0%</span></div>
              <div className="comp-status">STATUS: INCOMPLETE MATRIX</div>
            </div>
            <div className="comp-card amber" data-scroll data-scroll-speed="2.5">
              <div className="comp-header text-orange">ACRUX Fusion Engine</div>
              <div className="comp-row"><span className="label text-primary">Text + Audio Matrix</span><span className="val text-orange">99.9%</span></div>
              <div className="comp-row"><span className="label text-primary">Micro-expression Map</span><span className="val text-orange">94.5%</span></div>
              <div className="comp-row"><span className="label text-primary">Environmental Context</span><span className="val text-orange">100%</span></div>
              <div className="comp-status text-orange border-orange">STATUS: FULL SPECTRUM LOCK</div>
            </div>
          </div>
        </section>

        {/* CTA SECTION */}
        <section className="cta-section scroll-fade-up" data-scroll data-scroll-speed="1">
          <h2>Ready for the Future of Insight?</h2>
          <p>Integrate the ACRUX Fusion Engine into your data pipeline today. Experience semantic clarity without the noise.</p>
          <button className="cyber-btn primary mt-4" onClick={handleEnter}>
            INITIALIZE INTEGRATION
          </button>
        </section>
      </div>
    </>
  );
}

export default LandingPage;
