import { useEffect } from 'react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import LocomotiveScroll from 'locomotive-scroll';
import 'locomotive-scroll/dist/locomotive-scroll.css';

gsap.registerPlugin(ScrollTrigger);

export default function useLocoScroll(start) {
  useEffect(() => {
    if (!start) return;

    let locoScroll = null;
    const scrollEl = document.querySelector('[data-scroll-container]');

    locoScroll = new LocomotiveScroll({
      el: scrollEl,
      smooth: true,
      multiplier: 1,
      class: 'is-reveal'
    });

    // For Locomotive Scroll v5, native scroll is used, so we just setup a light integration
    // If you are using Locomotive v4, scrollerProxy would be here.
    // Locomotive v5 is generally natively compatible with ScrollTrigger.
    const lsUpdate = () => {
      if (locoScroll && locoScroll.update) {
        locoScroll.update();
      }
    };
    
    // Refresh ScrollTrigger and update LocomotiveScroll
    ScrollTrigger.addEventListener('refresh', lsUpdate);
    ScrollTrigger.refresh();

    return () => {
      if (locoScroll) {
        ScrollTrigger.removeEventListener('refresh', lsUpdate);
        if(locoScroll.destroy) locoScroll.destroy();
      }
    };
  }, [start]);
}
