import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

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
};

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  files: FileItem[] = [];

  selectedFile: FileItem | null = null;

  settings: Setting[] = [
    { key: 'name', label: 'Name', enabled: false },
    { key: 'address', label: 'Address', enabled: false },
    { key: 'phone', label: 'Phone', enabled: false },
    { key: 'email', label: 'Email', enabled: false }
  ];

  constructor(private sanitizer: DomSanitizer) {}

  onDragOver(event: DragEvent): void {
    event.preventDefault();
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();

    if (event.dataTransfer?.files) {
      this.addFiles(event.dataTransfer.files);
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;

    if (input.files) {
      this.addFiles(input.files);
      input.value = '';
    }
  }

  addFiles(fileList: FileList): void {
    const pdfFiles = Array.from(fileList).filter(
      (file) =>
        file.type === 'application/pdf' ||
        file.name.toLowerCase().endsWith('.pdf')
    );

    const newItems: FileItem[] = pdfFiles.map((file) => {
      const previewUrl = URL.createObjectURL(file);

      return {
        file,
        previewUrl,
        safePreviewUrl:
          this.sanitizer.bypassSecurityTrustResourceUrl(previewUrl),

        // placeholder output for now = original preview
        outputUrl: previewUrl,
        safeOutputUrl:
          this.sanitizer.bypassSecurityTrustResourceUrl(previewUrl)
      };
    });

    this.files.push(...newItems);

    if (!this.selectedFile && this.files.length > 0) {
      this.selectFile(this.files[0]);
    }
  }

  selectFile(fileItem: FileItem): void {
    this.selectedFile = fileItem;
  }

  removeFile(index: number): void {
    const removed = this.files[index];

    URL.revokeObjectURL(removed.previewUrl);

    if (removed.outputUrl !== removed.previewUrl) {
      URL.revokeObjectURL(removed.outputUrl);
    }

    this.files.splice(index, 1);

    if (this.selectedFile === removed) {
      this.selectedFile = this.files.length > 0 ? this.files[0] : null;
    }
  }

  downloadAllFiles(): void {
    if (this.files.length === 0) return;

    this.files.forEach((fileItem) => {
      const link = document.createElement('a');
      link.href = fileItem.outputUrl;
      link.download = this.getOutputFileName(fileItem.file.name);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  getOutputFileName(originalName: string): string {
    const lower = originalName.toLowerCase();

    if (lower.endsWith('.pdf')) {
      return originalName.slice(0, -4) + '_redacted.pdf';
    }

    return originalName + '_redacted.pdf';
  }
}