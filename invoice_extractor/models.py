"""Pydantic data models mirroring the sample pack's schema.json."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

NON_INVOICE_NULL_FIELDS = (
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "tax_total",
    "total",
)


class LineItem(BaseModel):
    """A single row from an invoice's item table."""

    model_config = ConfigDict(extra="forbid")

    description: str
    quantity: float | None
    amount: float


class InvoiceExtraction(BaseModel):
    """Structured extraction output matching schema.json."""

    model_config = ConfigDict(extra="forbid")

    is_invoice: bool
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    currency: str | None = Field(
        ...,
        pattern=r"^[A-Z]{3}$",
    )
    line_items: list[LineItem] | None
    tax_total: float | None
    total: float | None

    @model_validator(mode="after")
    def enforce_non_invoice_null_rule(self) -> InvoiceExtraction:
        """Require every invoice field to be null when is_invoice is false."""
        if not self.is_invoice:
            non_null = [
                field_name
                for field_name in NON_INVOICE_NULL_FIELDS
                if getattr(self, field_name) is not None
            ]
            if non_null:
                fields = ", ".join(non_null)
                raise ValueError(
                    f"Non-invoice documents must have null for all other fields; "
                    f"got non-null values for: {fields}"
                )
        return self
