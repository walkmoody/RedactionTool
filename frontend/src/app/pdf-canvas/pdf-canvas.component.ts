import {
  Component,
  Input,
  Output,
  EventEmitter,
  OnChanges,
  OnDestroy,
  ElementRef,
  ViewChild,
  AfterViewInit,
  NgZone,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import * as pdfjsLib from 'pdfjs-dist';

pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

@Component({
  selector: 'app-pdf-canvas',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './pdf-canvas.component.html',
  styleUrls: ['./pdf-canvas.component.css'],
})
export class PdfCanvasComponent implements OnChanges, OnDestroy, AfterViewInit {
  @Input() blobUrl: string = '';
  @Output() selectionChange = new EventEmitter<string>();

  @ViewChild('container') containerRef!: ElementRef<HTMLDivElement>;

  private rendering = false;
  private pendingRender = false;

  private selectionHandler = () => {
    const sel = window.getSelection()?.toString().trim() ?? '';
    this.zone.run(() => this.selectionChange.emit(sel));
  };

  constructor(private zone: NgZone) {}

  ngAfterViewInit(): void {
    document.addEventListener('selectionchange', this.selectionHandler);
    if (this.blobUrl) this.render();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['blobUrl'] && this.containerRef) this.render();
  }

  ngOnDestroy(): void {
    document.removeEventListener('selectionchange', this.selectionHandler);
  }

  private async render(): Promise<void> {
    if (this.rendering) {
      this.pendingRender = true;
      return;
    }

    this.rendering = true;
    this.pendingRender = false;
    const urlAtStart = this.blobUrl;
    const container = this.containerRef?.nativeElement;

    if (!container || !urlAtStart) {
      this.rendering = false;
      return;
    }

    container.innerHTML = '';

    try {
      const pdf = await pdfjsLib.getDocument(urlAtStart).promise;
      const dpr = window.devicePixelRatio || 1;

      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        if (this.blobUrl !== urlAtStart) break;

        const page = await pdf.getPage(pageNum);
        const containerWidth = container.clientWidth || container.offsetWidth || 600;
        const scale = containerWidth / page.getViewport({ scale: 1 }).width;
        const viewport = page.getViewport({ scale });

        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.width = `${viewport.width}px`;
        wrapper.style.height = `${viewport.height}px`;
        wrapper.style.marginBottom = '8px';
        wrapper.style.setProperty('--scale-factor', String(scale));
        wrapper.style.setProperty('--user-unit', '1');
        wrapper.style.setProperty('--total-scale-factor', String(scale));

        // Canvas — scaled for device pixel ratio to avoid blur
        const canvas = document.createElement('canvas');
        canvas.width = Math.floor(viewport.width * dpr);
        canvas.height = Math.floor(viewport.height * dpr);
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        canvas.style.display = 'block';
        wrapper.appendChild(canvas);

        const ctx = canvas.getContext('2d')!;
        ctx.scale(dpr, dpr);
        await page.render({ canvas, canvasContext: ctx, viewport }).promise;

        if (this.blobUrl !== urlAtStart) break;

        const textDiv = document.createElement('div');
        textDiv.className = 'textLayer';
        wrapper.appendChild(textDiv);

        const textLayer = new pdfjsLib.TextLayer({
          textContentSource: await page.getTextContent(),
          container: textDiv,
          viewport,
        });
        await textLayer.render();

        container.appendChild(wrapper);
      }
    } catch (e) {
      console.error('PDF render error:', e);
    } finally {
      this.rendering = false;
      if (this.pendingRender) this.render();
    }
  }
}
