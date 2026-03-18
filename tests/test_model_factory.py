import pytest

from main import AppSettings, ModelFactory, ModelRole, Provider, RoleModelConfig


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        planner=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
    )


def test_factory_builds_all_roles(settings: AppSettings) -> None:
    factory = ModelFactory(settings)
    for role in ModelRole:
        model = factory.create(role)
        assert model is not None
