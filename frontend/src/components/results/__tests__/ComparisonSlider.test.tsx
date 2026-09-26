import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ComparisonSlider } from '../ComparisonSlider'
import type { DecodedRaster } from '../../../lib/geotiff'

function raster(width: number, height: number): DecodedRaster {
  const band = () => new Float32Array(width * height).fill(0.2)
  return { width, height, bands: [band(), band(), band(), band()] }
}

describe('ComparisonSlider', () => {
  // Scenes are no longer always square (frame.tiling): the frame must take the raster's own
  // aspect ratio, or a rectangular scene would be stretched on screen.
  it('sizes its frame to a rectangular scene instead of forcing a square', () => {
    const { container } = render(<ComparisonSlider nativeRaster={raster(300, 200)} srRaster={raster(1200, 800)} />)
    const frame = container.querySelector('.comparison-slider__frame') as HTMLElement
    expect(frame.style.aspectRatio.replace(/\s/g, '')).toBe('1200/800')
  })

  it('still gives a square scene a square frame', () => {
    const { container } = render(<ComparisonSlider nativeRaster={raster(128, 128)} srRaster={raster(512, 512)} />)
    const frame = container.querySelector('.comparison-slider__frame') as HTMLElement
    expect(frame.style.aspectRatio.replace(/\s/g, '')).toBe('512/512')
  })
})
