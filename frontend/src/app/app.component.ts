import JSZip from 'jszip';
import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { PdfCanvasComponent } from './pdf-canvas/pdf-canvas.component';

type Setting = {
  key: string;
  label: string;
  enabled: boolean;
};

type FileItem = {
  file: File;
  previewUrl: string;
  safePreviewUrl: SafeResourceUrl;
  outputUrl: string;
  safeOutputUrl: SafeResourceUrl;
  status: 'idle' | 'loading' | 'done' | 'error';
};

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule, PdfCanvasComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  files = signal<FileItem[]>([]);
  selectedFile = signal<FileItem | null>(null);
  exclusions = '';
  manualSelection = signal<string>('');

  settings: Setting[] = [
    { key: 'PERSON', label: 'Name', enabled: true },
    { key: 'US_ADDRESS', label: 'Address', enabled: true },
    { key: 'PHONE_NUMBER', label: 'Phone', enabled: true },
    { key: 'EMAIL_ADDRESS', label: 'Email', enabled: true },
    { key: 'US_SSN', label: 'SSN', enabled: true },
    { key: 'CREDIT_CARD', label: 'Credit Card', enabled: true },
    { key: 'STRICT_DATE', label: 'DOB', enabled: true },
    { key: 'MEDICAL_RECORD_NUMBER', label: 'Medical Record Number', enabled: true },
    { key : 'INSURANCE', label: 'Insurance', enabled: true },
    { key: 'NPI', label: 'NPI', enabled: true }
  ];

  constructor(private sanitizer: DomSanitizer, private http: HttpClient) {}

  onDragOver(event: DragEvent): void { event.preventDefault(); }
  onDragLeave(event: DragEvent): void { event.preventDefault(); }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    if (event.dataTransfer?.files) this.addFiles(event.dataTransfer.files);
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      this.addFiles(input.files);
      input.value = '';
    }
  }

  addFiles(fileList: FileList): void {
    const newItems: FileItem[] = Array.from(fileList)
      .filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'))
      .map(file => {
        const previewUrl = URL.createObjectURL(file);
        return {
          file,
          previewUrl,
          safePreviewUrl: this.sanitizer.bypassSecurityTrustResourceUrl(previewUrl + '#toolbar=0&navpanes=0'),
          outputUrl: previewUrl,
          safeOutputUrl: this.sanitizer.bypassSecurityTrustResourceUrl(previewUrl + '#toolbar=0&navpanes=0'),
          status: 'idle' as const
        };
      });

    this.files.update(f => [...f, ...newItems]);

    if (!this.selectedFile() && this.files().length > 0) {
      this.selectedFile.set(this.files()[0]);
    }

    newItems.forEach(item => this.redactFile(item));
  }

  reRedact(): void {
    this.files().forEach(fileItem => {
      if (fileItem.status !== 'loading') this.redactFile(fileItem);
    });
  }

  redactFile(fileItem: FileItem): void {
    const selectedEntities = this.settings.filter(s => s.enabled).map(s => s.key);
    if (selectedEntities.length === 0) return;

    fileItem.status = 'loading';
    this.files.update(f => [...f]);

    const formData = new FormData();
    const excludedList = this.exclusions.split(',').map(s => s.trim()).filter(s => s.length > 0);
    formData.append('file', fileItem.file);
    formData.append('entities', JSON.stringify(selectedEntities));
    formData.append('exclusions', JSON.stringify(excludedList));

    this.http
      .post(`http://127.0.0.1:8000/api/redact?t=${Date.now()}`, formData, { responseType: 'blob' })
      .subscribe({
        next: (blob) => {
          if (fileItem.outputUrl !== fileItem.previewUrl) URL.revokeObjectURL(fileItem.outputUrl);
          fileItem.outputUrl = URL.createObjectURL(blob);
          fileItem.safeOutputUrl = this.sanitizer.bypassSecurityTrustResourceUrl(fileItem.outputUrl + '#toolbar=0&navpanes=0');
          fileItem.status = 'done';
          this.files.update(f => [...f]);
          this.syncSelected(fileItem);
        },
        error: (err) => {
          console.error('Redaction error:', err);
          fileItem.status = 'error';
          this.files.update(f => [...f]);
          this.syncSelected(fileItem);
        }
      });
  }

  private syncSelected(fileItem: FileItem): void {
    if (this.selectedFile()?.file.name === fileItem.file.name) {
      this.selectedFile.set({ ...fileItem });
    }
  }

  selectFile(fileItem: FileItem): void {
    this.selectedFile.set(fileItem);
  }

  cycleFile(dir: 1 | -1): void {
    const list = this.files();
    const idx = list.indexOf(this.selectedFile()!);
    const next = (idx + dir + list.length) % list.length;
    this.selectedFile.set(list[next]);
  }

  removeFile(index: number): void {
    const removed = this.files()[index];
    URL.revokeObjectURL(removed.previewUrl);
    if (removed.outputUrl !== removed.previewUrl) URL.revokeObjectURL(removed.outputUrl);
    this.files.update(f => f.filter((_, i) => i !== index));
    if (this.selectedFile() === removed) {
      this.selectedFile.set(this.files().length > 0 ? this.files()[0] : null);
    }
  }

  async downloadAllFiles(): Promise<void> {
    const doneFiles = this.files().filter(f => f.outputUrl);
    if (doneFiles.length === 1) {
      const link = document.createElement('a');
      link.href = doneFiles[0].outputUrl;
      link.download = this.getOutputFileName(doneFiles[0].file.name);
      link.click();
      return;
    }
    const zip = new JSZip();
    for (const fileItem of doneFiles) {
      const blob = await fetch(fileItem.outputUrl).then(r => r.blob());
      zip.file(this.getOutputFileName(fileItem.file.name), blob);
    }
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(zipBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'redacted_files.zip';
    link.click();
    URL.revokeObjectURL(url);
  }

  manualRedact(fileItem: FileItem): void {
    const term = this.manualSelection();
    if (!term) return;

    fetch(fileItem.outputUrl)
      .then(r => r.blob())
      .then(blob => {
        const formData = new FormData();
        formData.append('file', blob, fileItem.file.name);
        formData.append('terms', JSON.stringify([term]));

        fileItem.status = 'loading';
        this.files.update(f => [...f]);

        this.http
          .post(`http://127.0.0.1:8000/api/redact-terms?t=${Date.now()}`, formData, { responseType: 'blob' })
          .subscribe({
            next: (result) => {
              if (fileItem.outputUrl !== fileItem.previewUrl) URL.revokeObjectURL(fileItem.outputUrl);
              fileItem.outputUrl = URL.createObjectURL(result);
              fileItem.safeOutputUrl = this.sanitizer.bypassSecurityTrustResourceUrl(fileItem.outputUrl + '#toolbar=0&navpanes=0');
              fileItem.status = 'done';
              this.manualSelection.set('');
              this.files.update(f => [...f]);
              this.syncSelected(fileItem);
            },
            error: (err) => {
              console.error('Manual redaction error:', err);
              fileItem.status = 'done';
              this.files.update(f => [...f]);
            }
          });
      });
  }

  truncateName(name: string, max = 23): string {
    return name.length > max ? name.slice(0, max) + '...' : name;
  }

  getOutputFileName(name: string): string {
    return name.toLowerCase().endsWith('.pdf')
      ? name.slice(0, -4) + '_redacted.pdf'
      : name + '_redacted.pdf';
  }
}
